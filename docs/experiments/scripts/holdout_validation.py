"""Validacao fora da amostra dos grupos regionais (mesma amostragem do notebook 7.1).

Calibracao: 50 municipios sorteados por cluster (seed 42, celula 5 do notebook).
Holdout: todos os demais municipios do cluster, nunca vistos pelo otimizador.
Compara parametros calibrados x parametros padrao da variedade.
"""
import json, os, random, sys, time
ROOT = r"D:\_py\AgroIA_prod"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "util")); os.chdir(ROOT)
OUT = os.path.join(ROOT, "output", "soja_pr", "Optimization", os.environ.get("HOLDOUT_RUN", "v2_kgha_140d"))
HOLD_DIR = os.environ.get("HOLDOUT_DIR", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(HOLD_DIR, exist_ok=True)
REPORT = os.path.join(HOLD_DIR, "holdout_report.json")
MAX_HOLDOUT = int(os.environ.get("HOLDOUT_MAX_PER_CLUSTER", 40))


def metrics(df):
    import numpy as np
    df = df.dropna(subset=['dyield_obs', 'dyield_sim'])
    o, s = df.dyield_obs.values, df.dyield_sim.values
    return {'n': int(len(df)), 'obs_mean': float(o.mean()), 'sim_mean': float(s.mean()),
            'rmse_kg_ha': float(np.sqrt(np.mean((s - o) ** 2))), 'bias_kg_ha': float(np.mean(s - o)),
            'mae_kg_ha': float(np.mean(np.abs(s - o))), 'obs_sd': float(o.std()),
            'corr': float(np.corrcoef(o, s)[0, 1]) if s.std() > 0 else None,
            'r2_1to1': float(1 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2))}


def main():
    import nlopt, numpy as np, pandas as pd
    from util.SensitivityAnalyzer_soja_pr import SoyNetCDFDataLoader
    from util.NLOPT_regional_soja_pr import SoyWOFOSTRegionalOptimizerPR, skip_pcse_forced_gc
    from util.utils_soja_pr import setup_paths_soja_pr

    paths = setup_paths_soja_pr()
    loader = SoyNetCDFDataLoader(paths['COMPLETO'])
    opt = SoyWOFOSTRegionalOptimizerPR(paths, loader, {}, algorithm=nlopt.LN_BOBYQA, max_eval=10)

    random.seed(42)  # identico a celula 5 do notebook 7.1
    by = {}
    for f in loader.nc_files:
        by.setdefault(loader.get_point_info(f)['cluster_id'], []).append(f)
    grupos = {cid: random.sample(arq, min(50, len(arq))) for cid, arq in by.items()}

    report = {}
    rng = random.Random(7)
    t0 = time.time()
    with skip_pcse_forced_gc(True):
        for label, calib in sorted(grupos.items()):
            pfile = os.path.join(OUT, f"OPT_REGIONAL_group{label}_params.csv")
            if not os.path.exists(pfile):
                print(f"[pulado] grupo {label}: sem parametros", flush=True)
                continue
            params = pd.read_csv(pfile).iloc[0].drop(['group_label', 'n_municipios']).to_dict()
            hold = [f for f in by[label] if f not in set(calib)]
            hold = rng.sample(hold, min(MAX_HOLDOUT, len(hold)))
            frames_cal, frames_def = [], []
            for f in hold:
                info, wdf = loader.load_point_data(f)
                yd = opt.prepare_multiyear_context(info, wdf, info['point_id'])
                yd = [y for y in yd if np.isfinite(y['dyield_target']) and y['dyield_target'] > 0]
                if yd:
                    for frames, p in ((frames_cal, params), (frames_def, {})):
                        m = opt._calculate_yearly_metrics(p, yd)
                        m.insert(0, 'point_id', info['point_id'])
                        frames.append(m)
                for y in yd:
                    if os.path.exists(y['agro_temp_file']):
                        os.remove(y['agro_temp_file'])
            cal = pd.concat(frames_cal, ignore_index=True)
            dfl = pd.concat(frames_def, ignore_index=True)
            report[str(label)] = {'n_municipios_holdout': len(hold), 'calibrado': metrics(cal), 'padrao': metrics(dfl)}
            cal.to_csv(os.path.join(os.path.dirname(REPORT), f"holdout_group{label}.csv"), index=False)
            print(f"grupo {label}: {json.dumps(report[str(label)]['calibrado'])}  (t={time.time() - t0:.0f}s)", flush=True)
    json.dump(report, open(REPORT, "w"), indent=2)


if __name__ == "__main__":
    main()
