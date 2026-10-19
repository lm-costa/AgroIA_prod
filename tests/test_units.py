"""Unidades: IBGE/SIDRA (observado) e TWSO do PCSE (simulado) devem estar em kg/ha."""
import glob
import logging
import os

import numpy as np
import pytest

import NLOPT_soja_pr
from utils_soja_pr import to_dry_matter, SOYBEAN_MOISTURE_STANDARD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC_DIR = os.path.join(ROOT, "inputs", "data", "soja_pr", "completo")
HAS_NC = bool(glob.glob(os.path.join(NC_DIR, "point_id_*.nc")))


class _FakeWofost:
    """Stub do Wofost72_WLP_FD: TWSO em kg/ha, como o PCSE reporta."""

    def __init__(self, *args, **kwargs):
        pass

    def run_till_terminate(self):
        pass

    def get_output(self):
        return [{'TWSO': 5000.0}]


def _bare_optimizer():
    opt = object.__new__(NLOPT_soja_pr.SoyWOFOSTMultiYearOptimizerPR)
    opt._debug_failures_logged = 0
    opt.logger = logging.getLogger("test")
    opt._configure_parameter_tables = lambda model_params, parameters: None
    return opt


def test_simulated_yield_is_twso_in_kg_ha(monkeypatch):
    # PCSE reporta TWSO em kg/ha: o retorno em kg/ha nao pode ter fator extra.
    monkeypatch.setattr(NLOPT_soja_pr, "Wofost72_WLP_FD", _FakeWofost)
    assert _bare_optimizer().run_wofost_simulation({}, None, None, None) == pytest.approx(5000.0)


def test_to_dry_matter_factor():
    assert SOYBEAN_MOISTURE_STANDARD == 0.13
    assert to_dry_matter(3000.0) == pytest.approx(2610.0)
    assert to_dry_matter(0.0) == 0.0


@pytest.mark.skipif(not HAS_NC, reason="sem .nc de soja/PR")
def test_observed_yield_in_nc_is_kg_ha():
    # SIDRA 5457, variavel 112 = "Rendimento medio" em Quilogramas por Hectare.
    import xarray as xr
    values = []
    for f in sorted(glob.glob(os.path.join(NC_DIR, "point_id_*.nc")))[:20]:
        with xr.open_dataset(f) as ds:
            values.extend(v for v in ds['yield'].values if np.isfinite(v))
    median = float(np.median(values))
    assert 1000 < median < 6000, f"mediana {median}: nao parece kg/ha (t/ha seria ~3)"


@pytest.mark.skipif(not HAS_NC, reason="sem .nc de soja/PR")
def test_reference_simulation_is_agronomically_plausible():
    """Variedade padrao + solo ec4 + clima real: TWSO deve ficar na faixa de soja (kg/ha)."""
    from utils_soja_pr import setup_paths_soja_pr
    from SensitivityAnalyzer_soja_pr import SoyNetCDFDataLoader
    from NLOPT_regional_soja_pr import SoyWOFOSTRegionalOptimizerPR

    paths = setup_paths_soja_pr()
    loader = SoyNetCDFDataLoader(paths['COMPLETO'])
    opt = SoyWOFOSTRegionalOptimizerPR(paths, loader, {0.0: []})
    nc = sorted(glob.glob(os.path.join(NC_DIR, "point_id_*.nc")))[0]
    info, wdf = loader.load_point_data(nc)
    samples = opt.prepare_multiyear_context(info, wdf, "pytest_ref")
    try:
        assert samples
        sims = [opt.run_wofost_simulation({}, s['parameters'], s['weather'], s['agromanagement'])
                for s in samples[:3]]
    finally:
        for s in samples:
            if os.path.exists(s['agro_temp_file']):
                os.remove(s['agro_temp_file'])
    assert all(np.isfinite(sims))
    assert all(1000 < v < 9000 for v in sims), sims


@pytest.mark.skipif(not HAS_NC, reason="sem .nc de soja/PR")
def test_calibration_target_is_raw_observed_yield_not_detrended():
    """O alvo e a produtividade observada direta ('yield'), nao a sem tendencia ('dyield')."""
    import pandas as pd
    from utils_soja_pr import setup_paths_soja_pr, safra_ano_colheita
    from SensitivityAnalyzer_soja_pr import SoyNetCDFDataLoader
    from NLOPT_regional_soja_pr import SoyWOFOSTRegionalOptimizerPR

    assert NLOPT_soja_pr.SoyWOFOSTMultiYearOptimizerPR.OBSERVED_YIELD_COLUMN == 'yield'
    paths = setup_paths_soja_pr()
    loader = SoyNetCDFDataLoader(paths['COMPLETO'])
    opt = SoyWOFOSTRegionalOptimizerPR(paths, loader, {}, algorithm=0, max_eval=1)
    nc = sorted(glob.glob(os.path.join(NC_DIR, "point_id_*.nc")))[0]
    info, wdf = loader.load_point_data(nc)
    samples = opt.prepare_multiyear_context(info, wdf, "pytest_target")
    try:
        assert samples
        wdf = wdf.copy()
        wdf['safra'] = pd.to_datetime(wdf['date']).apply(safra_ano_colheita)
        raw = wdf.groupby('safra')['yield'].mean()
        det = wdf.groupby('safra')['dyield'].mean()
        for s in samples:
            assert s['dyield_target'] == pytest.approx(to_dry_matter(raw[s['year']]))
        # e a serie sem tendencia realmente difere em pelo menos uma safra
        assert any(abs(raw[s['year']] - det[s['year']]) > 1.0 for s in samples)
    finally:
        for s in samples:
            if os.path.exists(s['agro_temp_file']):
                os.remove(s['agro_temp_file'])
