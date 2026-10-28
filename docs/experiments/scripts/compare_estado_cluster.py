"""Compara estado(50), estado(100) e cluster em municipios fora de TODAS as amostras de calibracao."""
import os, sys, json, random
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "util")); os.chdir(ROOT)
S = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.join(ROOT, "output", "soja_pr", "Optimization")

def main():
    import nlopt, numpy as np, pandas as pd
    from util.SensitivityAnalyzer_soja_pr import SoyNetCDFDataLoader
    from util.NLOPT_regional_soja_pr import SoyWOFOSTRegionalOptimizerPR, skip_pcse_forced_gc
    from util.utils_soja_pr import setup_paths_soja_pr
    paths = setup_paths_soja_pr(); loader = SoyNetCDFDataLoader(paths['COMPLETO'])
    opt = SoyWOFOSTRegionalOptimizerPR(paths, loader, {}, nlopt.LN_BOBYQA, 10)
    prm = lambda f: pd.read_csv(f).iloc[0].drop(['group_label', 'n_municipios']).to_dict()
    pid = lambda f: os.path.basename(f)[len("point_id_"):-3]
    used = set()
    for tag, fn in (("v3_estado", "OPT_REGIONAL_groupPR_yearly_metrics.csv"), ("v3_estado100", "OPT_REGIONAL_groupPR_yearly_metrics.csv")):
        used |= set(pd.read_csv(os.path.join(OPT, tag, fn)).point_id.astype(str))
    for c in range(4):
        used |= set(pd.read_csv(os.path.join(OPT, "v3_yield_tabelas", f"OPT_REGIONAL_group{c}.0_yearly_metrics.csv")).point_id.astype(str))
    sets_state = {"estado50": prm(os.path.join(OPT, "v3_estado", "OPT_REGIONAL_groupPR_params.csv")),
                  "estado100": prm(os.path.join(OPT, "v3_estado100", "OPT_REGIONAL_groupPR_params.csv"))}
    by = {}
    for f in loader.nc_files:
        by.setdefault(loader.get_point_info(f)['cluster_id'], []).append(f)
    out = {k: [] for k in ("estado50", "estado100", "cluster")}; meta = {}
    with skip_pcse_forced_gc(True):
        for c in range(4):
            p_cl = prm(os.path.join(OPT, "v3_yield_tabelas", f"OPT_REGIONAL_group{c}.0_params.csv"))
            files = [f for f in by[float(c)] if pid(f) not in used]
            meta[c] = len(files); print(f"cluster {c}: {len(files)} municipios fora de todas as amostras", flush=True)
            fr = {k: [] for k in out}
            for f in files:
                info, wdf = loader.load_point_data(f)
                yd = [y for y in opt.prepare_multiyear_context(info, wdf, info['point_id']) if y['dyield_target'] > 0]
                for k, p in (("estado50", sets_state["estado50"]), ("estado100", sets_state["estado100"]), ("cluster", p_cl)):
                    m = opt._calculate_yearly_metrics(p, yd); m.insert(0, 'point_id', info['point_id']); m.insert(1, 'cluster', c); fr[k].append(m)
                for y in yd:
                    if os.path.exists(y['agro_temp_file']): os.remove(y['agro_temp_file'])
            for k in out:
                if fr[k]:
                    d = pd.concat(fr[k], ignore_index=True).dropna(subset=['dyield_obs', 'dyield_sim']); out[k].append(d)
                    os.makedirs(os.path.join(S, f"cmp3_{k}"), exist_ok=True); d.to_csv(os.path.join(S, f"cmp3_{k}", f"holdout_group{c}.0.csv"), index=False)
    rep = {}
    for k, frames in out.items():
        rep[k] = {}
        for d in frames + [pd.concat(frames, ignore_index=True)]:
            c = "todos" if d.cluster.nunique() > 1 else int(d.cluster.iloc[0])
            ao = d.dyield_obs - d.groupby('point_id').dyield_obs.transform('mean'); as_ = d.dyield_sim - d.groupby('point_id').dyield_sim.transform('mean')
            rep[k][str(c)] = dict(n=len(d), rmse=float(np.sqrt(((d.dyield_sim - d.dyield_obs) ** 2).mean())), bias=float((d.dyield_sim - d.dyield_obs).mean()),
                                  r=float(np.corrcoef(d.dyield_obs, d.dyield_sim)[0, 1]), r_temp=float(np.corrcoef(ao, as_)[0, 1]), sd_obs=float(d.dyield_obs.std()))
    json.dump({"municipios": meta, "resultado": rep}, open(os.path.join(S, "compare3.json"), "w"), indent=2)
    print(json.dumps(rep, indent=1))

if __name__ == "__main__":
    main()
