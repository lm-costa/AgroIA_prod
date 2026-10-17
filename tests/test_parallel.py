"""Avaliacao paralela das amostras: deve dar exatamente o mesmo objetivo que a serial."""
import glob
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC_DIR = os.path.join(ROOT, "inputs", "data", "soja_pr", "completo")
HAS_NC = len(glob.glob(os.path.join(NC_DIR, "point_id_*.nc"))) >= 3


@pytest.mark.skipif(not HAS_NC, reason="sem .nc de soja/PR")
def test_parallel_objective_matches_serial():
    import nlopt
    from utils_soja_pr import setup_paths_soja_pr
    from SensitivityAnalyzer_soja_pr import SoyNetCDFDataLoader
    from NLOPT_regional_soja_pr import SoyWOFOSTRegionalOptimizerPR

    paths = setup_paths_soja_pr()
    loader = SoyNetCDFDataLoader(paths['COMPLETO'])
    opt = SoyWOFOSTRegionalOptimizerPR(paths, loader, {}, algorithm=nlopt.LN_BOBYQA, max_eval=10)
    files = sorted(glob.glob(os.path.join(NC_DIR, "point_id_*.nc")))[:3]

    samples = opt.prepare_regional_samples(files)
    context = {'years_data': samples, 'param_names': ['TDWI', 'SPAN'], 'fixed_params': {}}
    points = [[80.0, 30.0], [120.0, 25.0], [60.0, 40.0]]  # varias avaliacoes seguidas
    try:
        serial = []
        for x in points:
            serial.append((opt.objective_function_regional(x, None, context), context['last_linear_rmse']))

        opt._start_workers(files, 2)
        try:
            parallel = []
            for x in points:
                parallel.append((opt.objective_function_regional(x, None, context), context['last_linear_rmse']))
        finally:
            opt._stop_workers()
    finally:
        for s in samples:
            if os.path.exists(s['agro_temp_file']):
                os.remove(s['agro_temp_file'])

    assert all(np.isfinite(v) and v < 1e9 for v, _ in serial)
    assert parallel == serial  # igualdade exata, nao aproximada


def test_skip_pcse_forced_gc_is_scoped_and_restored():
    import gc
    import pcse.engine
    from NLOPT_regional_soja_pr import skip_pcse_forced_gc

    original = pcse.engine.gc
    assert original is gc
    with skip_pcse_forced_gc(True):
        assert pcse.engine.gc is not gc
        assert pcse.engine.gc.collect() == 0
    assert pcse.engine.gc is gc
    with skip_pcse_forced_gc(False):
        assert pcse.engine.gc is gc
    assert pcse.engine.gc is gc


def test_worker_gc_skip_persists_after_python_gc():
    # Regressao: um context manager sem referencia era coletado e desfazia o bypass.
    import gc
    import pcse.engine
    from NLOPT_regional_soja_pr import _set_pcse_forced_gc_skipped

    previous = _set_pcse_forced_gc_skipped(True)
    try:
        gc.collect()
        assert pcse.engine.gc is not gc
        assert pcse.engine.gc.collect() == 0
    finally:
        pcse.engine.gc = previous
    assert pcse.engine.gc is gc
