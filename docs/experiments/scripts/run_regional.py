"""Executa as celulas de configuracao + otimizacao do notebook 7.1 sem interface.

Usa o proprio codigo do notebook (fonte unica): celulas 1, 3, 5, 7, 9 e 10.
Retomavel: checkpoints por etapa e resumos por grupo (ver optimize_group).
"""
import json, os, sys, time
ROOT = r"D:\_py\AgroIA_prod"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
CELLS = [1, 3, 5, 7, 9, 10]


def main():
    nb = json.load(open(os.path.join(ROOT, "7.1.Regional_Optimization_Soja_PR.ipynb"), encoding="utf-8"))
    ns = {"__name__": "__notebook__"}
    t0 = time.time()
    for i in CELLS:
        code = "".join(nb["cells"][i]["source"])
        print(f"\n##### celula {i} (t={time.time() - t0:.0f}s)", flush=True)
        exec(compile(code, f"<nb7.1 cell {i}>", "exec"), ns)
    print(f"\nFIM -- tempo total {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
