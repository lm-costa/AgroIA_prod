"""Diagnostico: sensibilidade da habilidade temporal do modelo a data de plantio (sem recalibrar)."""
import os, sys, random, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "util")); os.chdir(ROOT)
S = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.join(ROOT, "output", "soja_pr", "Optimization", "v3_yield_tabelas")
DATES = [(10, 1), (10, 15), (11, 1), (11, 15), (12, 1)]
N_MUN = int(os.environ.get("DIAG_N_MUN", 12))


def main():
    import nlopt, numpy as np, pandas as pd
    from util.SensitivityAnalyzer_soja_pr import SoyNetCDFDataLoader
    from util.NLOPT_regional_soja_pr import SoyWOFOSTRegionalOptimizerPR, skip_pcse_forced_gc
    from util.utils_soja_pr import setup_paths_soja_pr
    paths = setup_paths_soja_pr(); loader = SoyNetCDFDataLoader(paths['COMPLETO'])
    opt = SoyWOFOSTRegionalOptimizerPR(paths, loader, {}, nlopt.LN_BOBYQA, 10)
    prm = lambda f: pd.read_csv(f).iloc[0].drop(['group_label', 'n_municipios']).to_dict()
    random.seed(11)
    by = {}
    for f in loader.nc_files:
        by.setdefault(loader.get_point_info(f)['cluster_id'], []).append(f)
    files = {int(c): random.sample(v, min(N_MUN, len(v))) for c, v in sorted(by.items())}
    params = {c: prm(os.path.join(OPT, f"OPT_REGIONAL_group{c}.0_params.csv")) for c in files}
    rows = []
    with skip_pcse_forced_gc(True):
        for (m, d) in DATES:
            os.environ["SOJA_PR_SOWING_MONTH"], os.environ["SOJA_PR_SOWING_DAY"] = str(m), str(d)
            for c, fl in files.items():
                for f in fl:
                    info, wdf = loader.load_point_data(f)
                    yd = [y for y in opt.prepare_multiyear_context(info, wdf, info['point_id']) if y['dyield_target'] > 0]
                    for tag, p in (("padrao", {}), ("v3_cluster", params[c])):
                        if not yd: continue
                        mt = opt._calculate_yearly_metrics(p, yd)
                        mt['cluster'], mt['point_id'], mt['params'], mt['sow'] = c, info['point_id'], tag, f"{d:02d}/{m:02d}"
                        rows.append(mt)
                    for y in yd:
                        if os.path.exists(y['agro_temp_file']): os.remove(y['agro_temp_file'])
                print(f"plantio {d:02d}/{m:02d} cluster {c} ok", flush=True)
    df = pd.concat(rows, ignore_index=True).dropna(subset=['dyield_obs', 'dyield_sim'])
    df.to_csv(os.path.join(S, "sowing_diag.csv"), index=False)
    print("salvo sowing_diag.csv", len(df), flush=True)


if __name__ == "__main__":
    main()
