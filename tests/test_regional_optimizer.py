"""Otimizador regional: limites, falhas, teto de falha e persistencia."""
import json
import os

import numpy as np
import pytest

import NLOPT_regional_soja_pr as reg
import SensitivityAnalyzer
from utils import WOFOST_bounds


def _sample(i, target=3000.0):
    return {'year': 2016 + i, 'point_id': f"p{i}", 'parameters': None, 'weather': None,
            'agromanagement': None, 'dyield_target': target,
            'agro_temp_file': f"nao_existe_{i}.yaml"}


def _optimizer(tmp_path, sim, max_eval=60):
    import nlopt
    opt = object.__new__(reg.SoyWOFOSTRegionalOptimizerPR)
    opt.paths = {'OPTIMIZATION': str(tmp_path)}
    opt.algorithm = nlopt.LN_BOBYQA
    opt.max_eval = max_eval
    opt.prepare_regional_samples = lambda nc_files: [_sample(i) for i in range(5)]
    opt.run_wofost_simulation = sim
    return opt


def _linear_sim(model_params, *a, **k):
    # Modelo sintetico: produtividade = 30 * TDWI. Alvo 3000 -> TDWI otimo = 100.
    return 30.0 * model_params.get('TDWI', 50.0)


def test_bounds_are_ordered():
    names = ['TDWI', 'SPAN', 'TSUM1', 'TSUM2', 'DVSEND', 'AMAXTB000']
    for p, (lo, hi) in WOFOST_bounds(names).items():
        assert lo < hi, p


def test_all_stages_failing_is_reported_not_saved_as_valid(tmp_path):
    opt = _optimizer(tmp_path, lambda *a, **k: np.nan)
    result = opt.optimize_group(['a.nc'], 9.0, ['TDWI', 'SPAN'], max_params_per_stage=1, resume=False)
    assert result is not None and result.get('calibration_failed') is True
    assert not os.path.exists(tmp_path / "OPT_REGIONAL_group9.0_params.csv")
    summary = json.loads((tmp_path / "OPT_REGIONAL_group9.0_summary.json").read_text())
    assert summary['calibration_failed'] is True
    assert summary['rmse_aggregated'] is None  # JSON valido, sem Infinity


def test_successful_calibration_recovers_target(tmp_path):
    opt = _optimizer(tmp_path, _linear_sim)
    result = opt.optimize_group(['a.nc'], 1.0, ['TDWI'], max_params_per_stage=1, resume=False)
    assert not result.get('calibration_failed', False)
    assert result['optimized_params']['TDWI'] == pytest.approx(100.0, rel=0.02)
    assert np.isfinite(result['rmse_aggregated']) and result['rmse_aggregated'] < 100
    lo, hi = WOFOST_bounds(['TDWI'])['TDWI']
    assert lo <= result['optimized_params']['TDWI'] <= hi
    assert os.path.exists(tmp_path / "OPT_REGIONAL_group1.0_params.csv")


def test_failed_stage_keeps_previous_stage_params(tmp_path):
    def sim(model_params, *a, **k):
        if 'SPAN' in model_params:
            return np.nan
        return _linear_sim(model_params)

    opt = _optimizer(tmp_path, sim)
    result = opt.optimize_group(['a.nc'], 2.0, ['TDWI', 'SPAN'], max_params_per_stage=1, resume=False)
    assert 'SPAN' not in result['optimized_params']
    assert result['optimized_params']['TDWI'] == pytest.approx(100.0, rel=0.02)


def test_optimization_is_reproducible(tmp_path):
    out = []
    for name in ("r1", "r2"):
        d = tmp_path / name
        d.mkdir()
        res = _optimizer(d, _linear_sim).optimize_group(['a.nc'], 1.0, ['TDWI'], 1, resume=False)
        out.append(res['optimized_params']['TDWI'])
    assert out[0] == out[1]


def test_soja_sensitivity_analyzer_uses_kg_ha_directly():
    import SensitivityAnalyzer_soja_pr as sa
    assert sa.SoyMorrisScreeningAnalyzerPR.TWSO_TO_KG_HA == 1.0
    assert SensitivityAnalyzer.MorrisScreeningAnalyzer.TWSO_TO_KG_HA == 1000  # milho: inalterado
