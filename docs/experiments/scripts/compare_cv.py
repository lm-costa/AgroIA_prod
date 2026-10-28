"""
Comparacao pareada de duas validacoes cruzadas (mesmas particoes) feitas com cross_validation.py.

uso: python compare_cv.py <tag_a> <tag_b> [saida.json]

Para cada modo (espacial, temporal): metricas agregadas e por cluster de cada configuracao, o
modelo padrao da variedade como referencia e a diferenca (b - a) de RMSE, vies e r temporal com
IC 95% por bootstrap de MUNICIPIOS (reamostra municipios inteiros, preservando suas safras).
"""
import json, os, sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "util"))
from cv_utils_soja_pr import skill_metrics  # noqa: E402

OPT = os.path.join(ROOT, "output", "soja_pr", "Optimization")
N_BOOT = int(os.environ.get("CMP_BOOT", 300))


def _paired(a, b, rng):
    m = a.merge(b, on=["fold", "point_id", "year"], suffixes=("_a", "_b"))
    assert np.allclose(m.obs_a, m.obs_b), "observados diferentes: particoes nao sao as mesmas"
    groups = {p: g for p, g in m.groupby("point_id")}
    pts = np.array(list(groups))

    def met(df, col):
        r = skill_metrics(pd.DataFrame({"point_id": df.point_id, "obs": df.obs_a, "sim": df[col]}))
        return np.array([r["rmse"], r["bias"], r["r_temporal"]])

    base_a, base_b = met(m, "sim_a"), met(m, "sim_b")
    diffs = []
    for _ in range(N_BOOT):
        samp = rng.choice(pts, len(pts), replace=True)
        df = pd.concat([groups[p].assign(point_id=f"{p}_{i}") for i, p in enumerate(samp)], ignore_index=True)
        diffs.append(met(df, "sim_b") - met(df, "sim_a"))
    diffs = np.array(diffs)
    out = {}
    for i, name in enumerate(("rmse", "bias", "r_temporal")):
        lo, hi = np.percentile(diffs[:, i], [2.5, 97.5])
        out[name] = {"a": float(base_a[i]), "b": float(base_b[i]), "dif": float(base_b[i] - base_a[i]),
                     "ic95": [float(lo), float(hi)], "significativo": bool(lo > 0 or hi < 0)}
    return out


def main():
    tag_a, tag_b = sys.argv[1:3]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    rng = np.random.default_rng(0)
    sums = {t: json.load(open(os.path.join(OPT, f"cv_{t}", "cv_summary.json"))) for t in (tag_a, tag_b)}
    result = {}
    for kind in ("spatial", "temporal"):
        if not all(kind in s for s in sums.values()):
            continue
        print(f"\n######## CV {kind.upper()}")
        result[kind] = {"agregado": {}, "por_cluster": {}}
        for t in (tag_a, tag_b):
            m, p = sums[t][kind]["agregado_calibrado"], sums[t][kind]["agregado_padrao"]
            pp = sums[t][kind]["por_particao_calibrado"]
            print(f"[{t}] calibrado: RMSE {m['rmse']:.0f} | vies {m['bias']:+.0f} | r {m['r']:.2f} | r temporal {m['r_temporal']:.2f} "
                  f"| RMSE/sd {m['rmse_over_sd']:.2f} | n {m['n']}   (padrao: RMSE {p['rmse']:.0f}, r temporal {p['r_temporal']:.2f})")
            print(f"[{t}] entre particoes: RMSE {pp['rmse']['mean']:.0f} +- {pp['rmse']['sd']:.0f} | "
                  f"r temporal {pp['r_temporal']['mean']:.2f} +- {pp['r_temporal']['sd']:.2f}")
            result[kind]["agregado"][t] = {"calibrado": m, "padrao": p, "por_particao": pp}
        print("por cluster (a | b):")
        for c in "0123":
            a, b = sums[tag_a][kind]["por_cluster"][c], sums[tag_b][kind]["por_cluster"][c]
            print(f"  cluster {c}: RMSE {a['rmse']:.0f} | {b['rmse']:.0f}   vies {a['bias']:+.0f} | {b['bias']:+.0f}   "
                  f"r temp {a['r_temporal']:.2f} | {b['r_temporal']:.2f}   (n={a['n']}, sd obs {a['sd_obs']:.0f})")
            result[kind]["por_cluster"][c] = {tag_a: a, tag_b: b}
        pa = pd.read_csv(os.path.join(OPT, f"cv_{tag_a}", f"{kind}_predictions_all.csv"))
        pb = pd.read_csv(os.path.join(OPT, f"cv_{tag_b}", f"{kind}_predictions_all.csv"))
        paired = _paired(pa, pb, rng)
        result[kind]["pareado_b_menos_a"] = paired
        print(f"pareado ({tag_b} - {tag_a}), IC95% por bootstrap de municipios:")
        for name, v in paired.items():
            print(f"  {name:10s}: {v['a']:8.3f} -> {v['b']:8.3f} | dif {v['dif']:+8.3f}  IC95% [{v['ic95'][0]:+.3f}, {v['ic95'][1]:+.3f}]  "
                  f"({'significativo' if v['significativo'] else 'nao significativo'})")
    if out_path:
        json.dump(result, open(out_path, "w"), indent=2)
        print("salvo", out_path)


if __name__ == "__main__":
    main()
