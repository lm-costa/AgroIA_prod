"""
Particoes e metricas da validacao cruzada da calibracao regional (soja/PR).

Funcoes puras (sem WOFOST), para poderem ser testadas e reaproveitadas:
- make_spatial_folds: k partições de MUNICIPIOS, estratificadas por cluster;
- make_year_blocks: blocos contiguos de safras para o teste temporal;
- stratified_sample: amostra de treino proporcional ao tamanho de cada cluster;
- skill_metrics: RMSE, vies, r, r temporal (anomalia dentro do municipio) e RMSE/sd.
"""
import random

import numpy as np
import pandas as pd


def make_spatial_folds(files_by_cluster, k, seed):
    """
    {cluster: [arquivos]} -> lista de k conjuntos de arquivos (teste de cada
    partição). Cada municipio cai em exatamente uma partição, e cada cluster
    fica distribuido (quase) uniformemente entre as partições.
    """
    folds = [[] for _ in range(k)]
    rng = random.Random(seed)
    for cluster in sorted(files_by_cluster):
        files = sorted(files_by_cluster[cluster])
        rng.shuffle(files)
        for i, f in enumerate(files):
            folds[i % k].append(f)
    return folds


def make_year_blocks(years, n_blocks):
    """Safras ordenadas -> n_blocos contiguos de tamanho (quase) igual."""
    years = sorted(years)
    size, extra = divmod(len(years), n_blocks)
    blocks, start = [], 0
    for b in range(n_blocks):
        end = start + size + (1 if b < extra else 0)
        blocks.append(years[start:end])
        start = end
    return blocks


def stratified_sample(files_by_cluster, n_total, seed):
    """Amostra de ~n_total arquivos, proporcional ao tamanho de cada cluster (minimo 1)."""
    rng = random.Random(seed)
    total = sum(len(v) for v in files_by_cluster.values())
    picked = []
    for cluster in sorted(files_by_cluster):
        files = sorted(files_by_cluster[cluster])
        n = min(len(files), max(1, round(n_total * len(files) / total)))
        picked.extend(rng.sample(files, n))
    return picked


def skill_metrics(df, obs='obs', sim='sim', unit='point_id'):
    """
    Metricas de habilidade em kg/ha. r_temporal = correlacao das anomalias
    (valor - media do municipio), isolando a resposta entre safras; so usa
    municipios com >= 2 safras.
    """
    df = df.dropna(subset=[obs, sim])
    if len(df) < 3:
        return {'n': int(len(df))}
    o, s = df[obs].to_numpy(float), df[sim].to_numpy(float)
    ao = df[obs] - df.groupby(unit)[obs].transform('mean')
    asim = df[sim] - df.groupby(unit)[sim].transform('mean')
    multi = df.groupby(unit)[obs].transform('size') >= 2

    def corr(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float('nan')

    sd = float(o.std())
    rmse = float(np.sqrt(np.mean((s - o) ** 2)))
    return {
        'n': int(len(df)), 'municipios': int(df[unit].nunique()),
        'obs_mean': float(o.mean()), 'sim_mean': float(s.mean()),
        'rmse': rmse, 'bias': float(np.mean(s - o)), 'r': corr(o, s),
        'r_temporal': corr(ao[multi], asim[multi]), 'sd_obs': sd,
        'rmse_over_sd': rmse / sd if sd > 0 else float('nan'),
    }


def summarize_folds(fold_metrics):
    """Lista de dicts de metricas (uma por partição) -> media e desvio entre partições."""
    df = pd.DataFrame(fold_metrics)
    out = {}
    for col in ('rmse', 'bias', 'r', 'r_temporal', 'rmse_over_sd'):
        if col in df:
            out[col] = {'mean': float(df[col].mean()), 'sd': float(df[col].std(ddof=1)) if len(df) > 1 else float('nan')}
    return out
