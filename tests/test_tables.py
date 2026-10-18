"""Tabelas do WOFOST: calibrar alguns pontos nao pode descartar os demais pontos da variedade."""
import glob
import os

import numpy as np
import pytest

import NLOPT_soja_pr
from NLOPT_soja_pr import SoyWOFOSTMultiYearOptimizerPR as Soy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC_DIR = os.path.join(ROOT, "inputs", "data", "soja_pr", "completo")
HAS_NC = bool(glob.glob(os.path.join(NC_DIR, "point_id_*.nc")))

AMAXTB_DEFAULT = [0.00, 37.0, 1.70, 37.0, 2.00, 0.0]
TMPFTB_DEFAULT = [0.0, 0.0, 10.0, 0.0, 24.0, 1.0, 39.0, 1.0, 42.0, 0.63]


class _FakeParams:
    """Imita o ParameterProvider: tabelas padrao em _cropdata, overrides gravados."""

    def __init__(self, cropdata):
        self._cropdata = cropdata
        self.overrides = {}

    def set_override(self, name, value):
        self.overrides[name] = value


def _bare():
    opt = object.__new__(Soy)
    return opt


def _table(values):
    a = np.array(values, float)
    return a[0::2], a[1::2]


def test_uncalibrated_tables_are_left_untouched():
    p = _FakeParams({'AMAXTB': AMAXTB_DEFAULT})
    _bare()._configure_parameter_tables({'TDWI': 70.0}, p)
    assert 'AMAXTB' not in p.overrides and p.overrides['TDWI'] == 70.0


def test_partial_table_keeps_default_at_other_points():
    p = _FakeParams({'AMAXTB': AMAXTB_DEFAULT})
    _bare()._configure_parameter_tables({'AMAXTB082': 30.0}, p)
    xs, ys = _table(p.overrides['AMAXTB'])
    assert list(xs) == sorted(xs)
    assert dict(zip(xs, ys))[0.82] == 30.0                 # ponto calibrado
    assert dict(zip(xs, ys))[1.7] == 37.0                  # ponto original preservado
    assert dict(zip(xs, ys))[0.0] == 37.0 and dict(zip(xs, ys))[2.0] == 0.0  # padrao nos demais


def test_calibrated_knot_at_default_value_reproduces_default_curve():
    # Tabela inteira == padrao quando os joelhos calibrados valem o padrao interpolado.
    p = _FakeParams({'TMPFTB': TMPFTB_DEFAULT})
    dflt = lambda x: float(np.interp(x, *_table(TMPFTB_DEFAULT)))
    _bare()._configure_parameter_tables({'TMPFTB020': dflt(20.0), 'TMPFTB035': dflt(35.0)}, p)
    xs, ys = _table(p.overrides['TMPFTB'])
    for x in (0, 5, 10, 15, 20, 24, 30, 35, 39, 42, 45):
        assert np.interp(x, xs, ys) == pytest.approx(np.interp(x, *_table(TMPFTB_DEFAULT)))


def test_table_keys_are_not_forwarded_as_partial_tables():
    p = _FakeParams({'SLATB': [0.0, 0.0014, 0.45, 0.0025, 0.9, 0.0025, 2.0, 0.0007]})
    _bare()._configure_parameter_tables({'SLATB100': 0.003, 'SPAN': 25.0}, p)
    xs, ys = _table(p.overrides['SLATB'])
    assert len(xs) == 5  # 4 originais + o joelho 1.0 inserido
    assert p.overrides['SPAN'] == 25.0


@pytest.mark.skipif(not HAS_NC, reason="sem .nc de soja/PR")
def test_real_simulation_unchanged_when_knots_equal_default():
    import nlopt
    from utils_soja_pr import setup_paths_soja_pr
    from SensitivityAnalyzer_soja_pr import SoyNetCDFDataLoader
    from NLOPT_regional_soja_pr import SoyWOFOSTRegionalOptimizerPR

    paths = setup_paths_soja_pr()
    loader = SoyNetCDFDataLoader(paths['COMPLETO'])
    opt = SoyWOFOSTRegionalOptimizerPR(paths, loader, {}, algorithm=nlopt.LN_BOBYQA, max_eval=1)
    nc = sorted(glob.glob(os.path.join(NC_DIR, "point_id_*.nc")))[0]
    info, wdf = loader.load_point_data(nc)
    samples = opt.prepare_multiyear_context(info, wdf, "pytest_tables")[:2]
    try:
        for s in samples:
            cd = s['parameters']._cropdata
            dflt = lambda name, x: float(np.interp(x, *_table(cd[name])))
            knots = {'TMPFTB020': dflt('TMPFTB', 20), 'TMPFTB035': dflt('TMPFTB', 35),
                     'AMAXTB082': dflt('AMAXTB', 0.82), 'AMAXTB200': dflt('AMAXTB', 2.0),
                     'SLATB100': dflt('SLATB', 1.0)}
            s['parameters'].clear_override()
            base = opt.run_wofost_simulation({}, s['parameters'], s['weather'], s['agromanagement'])
            s['parameters'].clear_override()
            with_knots = opt.run_wofost_simulation(knots, s['parameters'], s['weather'], s['agromanagement'])
            assert with_knots == pytest.approx(base, rel=1e-6)
    finally:
        for s in samples:
            if os.path.exists(s['agro_temp_file']):
                os.remove(s['agro_temp_file'])
