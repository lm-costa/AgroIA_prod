"""Funcao objetivo regional: RMSE log1p, falhas e valores nao finitos."""
import numpy as np
import pytest

import NLOPT_regional_soja_pr as reg


def _optimizer(sim_values):
    """Otimizador sem __init__; run_wofost_simulation devolve sim_values em ordem."""
    opt = object.__new__(reg.SoyWOFOSTRegionalOptimizerPR)
    it = iter(sim_values)
    opt.run_wofost_simulation = lambda *a, **k: next(it)
    return opt


def _context(targets):
    return {
        'param_names': ['TDWI'],
        'fixed_params': {},
        'years_data': [
            {'parameters': None, 'weather': None, 'agromanagement': None, 'dyield_target': t}
            for t in targets
        ],
    }


def test_log1p_and_linear_rmse():
    targets = [2000.0, 3000.0, 4000.0]
    sims = [2500.0, 2500.0, 4500.0]
    ctx = _context(targets)
    value = _optimizer(sims).objective_function_regional([50.0], None, ctx)
    expected_log = np.sqrt(np.mean((np.log1p(sims) - np.log1p(targets)) ** 2))
    expected_lin = np.sqrt(np.mean((np.array(sims) - np.array(targets)) ** 2))
    assert value == pytest.approx(expected_log)
    assert ctx['last_linear_rmse'] == pytest.approx(expected_lin)


def test_perfect_fit_is_zero():
    ctx = _context([2000.0, 3000.0])
    assert _optimizer([2000.0, 3000.0]).objective_function_regional([1.0], None, ctx) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -5.0, 0.0, reg.MIN_PLAUSIBLE_YIELD - 1])
def test_invalid_simulation_counts_as_failure(bad):
    # 1 invalida em 3 (33%) > FAILURE_TOLERANCE -> penalidade, nunca NaN/inf.
    ctx = _context([2000.0, 3000.0, 4000.0])
    value = _optimizer([2000.0, bad, 4000.0]).objective_function_regional([1.0], None, ctx)
    assert value == 1e10
    assert ctx['last_linear_rmse'] is None


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_invalid_within_tolerance_is_excluded_not_propagated(bad):
    targets = [2000.0 + 100 * i for i in range(10)]
    sims = list(targets)
    sims[3] = bad  # 10% <= FAILURE_TOLERANCE
    ctx = _context(targets)
    value = _optimizer(sims).objective_function_regional([1.0], None, ctx)
    assert np.isfinite(value) and value == pytest.approx(0.0)
    assert np.isfinite(ctx['last_linear_rmse'])


def test_nonfinite_target_is_not_propagated():
    ctx = _context([2000.0, np.nan, 4000.0])
    value = _optimizer([2000.0, 3000.0, 4000.0]).objective_function_regional([1.0], None, ctx)
    assert value == 1e10 or np.isfinite(value)


def test_empty_samples_is_penalized():
    assert _optimizer([]).objective_function_regional([1.0], None, _context([])) == 1e10


def test_failure_never_beats_a_valid_fit():
    targets = [3000.0] * 5
    good = _optimizer([2800.0] * 5).objective_function_regional([1.0], None, _context(targets))
    failed = _optimizer([np.nan] * 5).objective_function_regional([1.0], None, _context(targets))
    assert failed > good
