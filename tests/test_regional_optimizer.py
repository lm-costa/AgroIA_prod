"""Otimizador regional: limites, falhas, teto de falha e persistencia."""
import json
import os

import numpy as np
import pytest

import util.NLOPT_regional_soja_pr as reg
import util.SensitivityAnalyzer as SensitivityAnalyzer
from util.utils import WOFOST_bounds


def _sample(i, target=3000.0):
    return {'year': 2016 + i, 'point_id': f"p{i}", 'parameters': None, 'weather': None,
            'agromanagement': None, 'target_yield': target,
            'agro_temp_file': f"nao_existe_{i}.yaml"}


SPREAD_TARGETS = [1500.0, 2500.0, 3500.0, 4500.0, 2000.0]  # RMSE minimo >> limiar de parada antecipada


def _optimizer(tmp_path, sim, max_eval=60, targets=None):
    import nlopt
    opt = object.__new__(reg.SoyWOFOSTRegionalOptimizerPR)
    opt.paths = {'OPTIMIZATION': str(tmp_path)}
    opt.algorithm = nlopt.LN_BOBYQA
    opt.max_eval = max_eval
    targets = targets or [3000.0] * 5
    opt.prepare_regional_samples = lambda nc_files, years=None: [_sample(i, t) for i, t in enumerate(targets)]
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
    import util.SensitivityAnalyzer_soja_pr as sa
    assert sa.SoyMorrisScreeningAnalyzerPR.TWSO_TO_KG_HA == 1.0
    assert SensitivityAnalyzer.MorrisScreeningAnalyzer.TWSO_TO_KG_HA == 1.0  # PCSE ja reporta kg/ha (milho tambem)


def _spy_dims(monkeypatch):
    import nlopt
    dims = []
    orig = nlopt.opt

    def spy(alg, n):
        dims.append(n)
        return orig(alg, n)

    monkeypatch.setattr(nlopt, 'opt', spy)
    return dims


def test_each_stage_searches_only_new_params(tmp_path, monkeypatch):
    # Sem dimensoes mortas: etapa k otimiza so os parametros novos.
    dims = _spy_dims(monkeypatch)
    opt = _optimizer(tmp_path, _linear_sim, targets=SPREAD_TARGETS)
    opt.optimize_group(['a.nc'], 3.0, ['TDWI', 'SPAN', 'TSUM1', 'TSUM2'], max_params_per_stage=2, resume=False)
    assert dims == [2, 2]


def test_previous_params_stay_frozen_in_later_stages(tmp_path):
    seen = []

    def sim(model_params, *a, **k):
        seen.append(dict(model_params))
        return _linear_sim(model_params)

    opt = _optimizer(tmp_path, sim, targets=SPREAD_TARGETS)
    res = opt.optimize_group(['a.nc'], 4.0, ['TDWI', 'SPAN'], max_params_per_stage=1, resume=False)
    tdwi = res['optimized_params']['TDWI']
    stage2 = [m for m in seen if 'SPAN' in m and m.get('TDWI') is not None and 'SPAN' in m]
    assert stage2, "etapa 2 nao rodou"
    assert all(m['TDWI'] == tdwi for m in stage2)


def test_params_of_discarded_stage_are_retried(tmp_path, monkeypatch):
    # SPAN sempre falha: etapa 2 e descartada; etapa 3 volta a tentar SPAN junto com TSUM1.
    dims = _spy_dims(monkeypatch)

    def sim(model_params, *a, **k):
        if 'SPAN' in model_params:
            return np.nan
        return _linear_sim(model_params)

    opt = _optimizer(tmp_path, sim, targets=SPREAD_TARGETS)
    res = opt.optimize_group(['a.nc'], 5.0, ['TDWI', 'SPAN', 'TSUM1'], max_params_per_stage=1, resume=False)
    assert dims == [1, 1, 2]
    assert 'SPAN' not in res['optimized_params']


def _raising_sim(model_params, *a, **k):
    if 'SPAN' in model_params:
        raise RuntimeError("falha simulada (ex.: processo trabalhador morreu)")
    return _linear_sim(model_params)


def test_error_in_a_stage_marks_result_incomplete_and_keeps_checkpoint(tmp_path):
    opt = _optimizer(tmp_path, _raising_sim, targets=SPREAD_TARGETS)
    res = opt.optimize_group(['a.nc'], 6.0, ['TDWI', 'SPAN'], max_params_per_stage=1, resume=True)
    assert res['incomplete'] is True and res['stage_error']['stage'] == 2
    assert set(res['optimized_params']) == {'TDWI'}  # etapa 1 aproveitada, mas marcada parcial
    summary = json.loads((tmp_path / "OPT_REGIONAL_group6.0_summary.json").read_text())
    assert summary['incomplete'] is True and summary['stage_error']['stage'] == 2
    assert os.path.exists(tmp_path / "OPT_REGIONAL_group6.0_checkpoint.json")
    # parcial nunca usa o nome dos arquivos validos (scripts leem esses CSVs direto)
    assert os.path.exists(tmp_path / "OPT_REGIONAL_group6.0_params_partial.csv")
    assert not os.path.exists(tmp_path / "OPT_REGIONAL_group6.0_params.csv")


def test_incomplete_run_resumes_from_failed_stage_and_completes(tmp_path):
    first = _optimizer(tmp_path, _raising_sim, targets=SPREAD_TARGETS)
    first.optimize_group(['a.nc'], 7.0, ['TDWI', 'SPAN'], max_params_per_stage=1, resume=True)

    seen = []

    def sim(model_params, *a, **k):
        seen.append(set(model_params))
        return _linear_sim(model_params)

    second = _optimizer(tmp_path, sim, targets=SPREAD_TARGETS)
    res = second.optimize_group(['a.nc'], 7.0, ['TDWI', 'SPAN'], max_params_per_stage=1, resume=True)
    assert res['incomplete'] is False and set(res['optimized_params']) == {'TDWI', 'SPAN'}
    assert all('SPAN' in names for names in seen)  # etapa 1 nao foi refeita
    assert not os.path.exists(tmp_path / "OPT_REGIONAL_group7.0_checkpoint.json")


def test_checkpoint_of_another_configuration_is_not_resumed(tmp_path):
    def interrupted(max_eval, years=None):
        opt = _optimizer(tmp_path, _raising_sim, max_eval=max_eval, targets=SPREAD_TARGETS)
        opt.optimize_group(['a.nc'], 8.0, ['TDWI', 'SPAN'], max_params_per_stage=1, resume=True, years=years)

    interrupted(max_eval=60)
    seen = []

    def sim(model_params, *a, **k):
        seen.append(set(model_params))
        return _linear_sim(model_params)

    # Outro orcamento de avaliacoes (outra configuracao): refaz desde a etapa 1.
    other = _optimizer(tmp_path, sim, max_eval=80, targets=SPREAD_TARGETS)
    other.optimize_group(['a.nc'], 8.0, ['TDWI', 'SPAN'], max_params_per_stage=1, resume=True)
    assert {'TDWI'} in seen


def test_config_signature_changes_with_files_years_calendar_and_budget(tmp_path, monkeypatch):
    base = _optimizer(tmp_path, _linear_sim).config_signature(['a.nc', 'b.nc'], None)
    assert base == _optimizer(tmp_path, _linear_sim).config_signature(['b.nc', 'a.nc'], None)  # ordem nao importa
    assert base != _optimizer(tmp_path, _linear_sim).config_signature(['a.nc'], None)
    assert base != _optimizer(tmp_path, _linear_sim).config_signature(['a.nc', 'b.nc'], {2016, 2017})
    assert base != _optimizer(tmp_path, _linear_sim, max_eval=61).config_signature(['a.nc', 'b.nc'], None)
    monkeypatch.setenv('SOJA_PR_SOWING_MONTH', '10')
    assert base != _optimizer(tmp_path, _linear_sim).config_signature(['a.nc', 'b.nc'], None)
