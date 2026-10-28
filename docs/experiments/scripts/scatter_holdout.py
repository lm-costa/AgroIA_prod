"""Scatter observado x simulado por cluster: calibracao (linha de cima) e validacao fora da amostra (baixo).

uso: scatter_holdout.py <saida.png> <titulo> <rotulo=pasta_com_holdout_group*.csv> [...]
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_png, title = sys.argv[1:3]
rows_spec = [a.split('=', 1) for a in sys.argv[3:]]  # rotulo=pasta
SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
LIM = (0, 5600)
clusters = [0, 1, 2, 3]

fig, axes = plt.subplots(len(rows_spec), 4, figsize=(15, 4.1 * len(rows_spec)), facecolor=SURFACE, sharex=True, sharey=True)
for row, (label, folder) in enumerate(rows_spec):
    loader = (lambda f: (lambda c: pd.read_csv(os.path.join(f, f"holdout_group{c}.0.csv"))))(folder)
    for col, c in enumerate(clusters):
        ax = axes[row, col]
        ax.set_facecolor(SURFACE)
        try:
            d = loader(c).dropna(subset=["dyield_obs", "dyield_sim"])
        except FileNotFoundError:
            ax.text(0.5, 0.5, "sem dados", transform=ax.transAxes, ha="center", color=INK2); continue
        o, s = d.dyield_obs.values, d.dyield_sim.values
        ax.plot(LIM, LIM, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.scatter(o, s, s=16, color=SERIES, alpha=0.45, edgecolors="none", zorder=2)
        rmse = np.sqrt(np.mean((s - o) ** 2)); bias = np.mean(s - o)
        r = np.corrcoef(o, s)[0, 1] if s.std() > 0 else float("nan")
        ax.text(0.04, 0.96, f"n = {len(d)}\nRMSE = {rmse:.0f}\nviés = {bias:+.0f}\nr = {r:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10, color=INK, linespacing=1.4)
        ax.set_xlim(LIM); ax.set_ylim(LIM); ax.set_aspect("equal", adjustable="box")
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        if row == 0: ax.set_title(f"Cluster {c}", color=INK, fontsize=12, loc="left")
        if col == 0: ax.set_ylabel(f"{label}\nsimulado (kg/ha)", color=INK2, fontsize=10)
        if row == len(rows_spec) - 1: ax.set_xlabel("observado (kg/ha)", color=INK2, fontsize=10)
fig.suptitle(title, color=INK, fontsize=14, x=0.06, ha="left", y=0.995)
fig.text(0.06, 0.005, "Tracejado: linha 1:1. RMSE e viés em kg/ha (matéria seca). Cada ponto = município × safra.",
         color=INK2, fontsize=9, ha="left")
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
fig.savefig(out_png, dpi=130, facecolor=SURFACE)
print("salvo", out_png)
