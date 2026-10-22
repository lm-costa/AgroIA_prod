import numpy as np
import pandas as pd
import pytest

from cv_utils_soja_pr import (make_spatial_folds, make_year_blocks, skill_metrics,
                              stratified_sample, summarize_folds)

FILES = {0.0: [f"a{i}" for i in range(23)], 1.0: [f"b{i}" for i in range(9)], 2.0: [f"c{i}" for i in range(7)]}


def test_spatial_folds_are_disjoint_complete_and_stratified():
    folds = make_spatial_folds(FILES, 5, seed=1)
    flat = [f for fold in folds for f in fold]
    assert sorted(flat) == sorted(f for v in FILES.values() for f in v)
    assert len(flat) == len(set(flat))
    for cluster, files in FILES.items():
        counts = [sum(f in files for f in fold) for fold in folds]
        assert max(counts) - min(counts) <= 1


def test_spatial_folds_are_reproducible_and_seed_dependent():
    assert make_spatial_folds(FILES, 5, 1) == make_spatial_folds(FILES, 5, 1)
    assert make_spatial_folds(FILES, 5, 1) != make_spatial_folds(FILES, 5, 2)


def test_year_blocks_cover_all_years_in_order():
    blocks = make_year_blocks(range(2016, 2025), 3)
    assert blocks == [[2016, 2017, 2018], [2019, 2020, 2021], [2022, 2023, 2024]]
    assert make_year_blocks([2016, 2017, 2018, 2019], 3) == [[2016, 2017], [2018], [2019]]


def test_stratified_sample_is_proportional_and_reproducible():
    s = stratified_sample(FILES, 20, seed=3)
    assert s == stratified_sample(FILES, 20, seed=3)
    assert len(set(s)) == len(s)
    n0 = sum(f.startswith("a") for f in s)
    assert n0 > sum(f.startswith("b") for f in s) > 0


def test_skill_metrics_perfect_and_biased():
    df = pd.DataFrame({'point_id': ['p'] * 4 + ['q'] * 4, 'obs': [1000., 2000, 3000, 4000] * 2})
    df['sim'] = df['obs']
    m = skill_metrics(df)
    assert m['rmse'] == pytest.approx(0) and m['r'] == pytest.approx(1) and m['r_temporal'] == pytest.approx(1)
    df['sim'] = df['obs'] - 100
    m = skill_metrics(df)
    assert m['bias'] == pytest.approx(-100) and m['rmse'] == pytest.approx(100)


def test_skill_metrics_temporal_isolates_year_response():
    # Municipios com niveis diferentes mas SEM resposta entre safras no simulado: r alto, r temporal ~ 0.
    rows = []
    for p, level in (('p', 1000.), ('q', 3000.)):
        for y, anomaly in zip(range(4), (-200., 200., 100., -100.)):
            rows.append({'point_id': p, 'obs': level + anomaly, 'sim': level})
    m = skill_metrics(pd.DataFrame(rows))
    assert m['r'] > 0.95
    assert np.isnan(m['r_temporal']) or abs(m['r_temporal']) < 1e-9


def test_summarize_folds_mean_and_sd():
    s = summarize_folds([{'rmse': 400, 'r': 0.5}, {'rmse': 500, 'r': 0.7}])
    assert s['rmse']['mean'] == 450 and s['rmse']['sd'] == pytest.approx(70.71, rel=1e-3)
