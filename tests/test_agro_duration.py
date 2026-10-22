"""max_duration deve ter uma unica fonte: map_info_soja_pr -> janela de clima e .agro."""
import datetime
import os
import shutil

import pytest
import yaml

from utils_soja_pr import map_info_soja_pr, update_agro_management_file_soja_pr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGRO = os.path.join(ROOT, "inputs", "data", "agro", "agro_soybean_pr.agro")


def _calendar(path):
    data = yaml.safe_load(open(path))
    return next(iter(data['AgroManagement'][0].values()))['CropCalendar']


def test_default_duration_is_140(monkeypatch):
    monkeypatch.delenv("SOJA_PR_MAX_DURATION", raising=False)
    assert map_info_soja_pr()['max_duration'] == 140


@pytest.mark.parametrize("duration", [120, 140, 160])
def test_agro_max_duration_follows_map_info(tmp_path, monkeypatch, duration):
    monkeypatch.setenv("SOJA_PR_MAX_DURATION", str(duration))
    agro = tmp_path / "a.agro"
    shutil.copy(AGRO, agro)
    update_agro_management_file_soja_pr(str(agro), datetime.datetime(2019, 11, 1))
    cal = _calendar(str(agro))
    assert cal['max_duration'] == duration
    assert cal['crop_end_date'] == datetime.date(2019, 11, 1) + datetime.timedelta(days=duration)


def test_default_sowing_is_nov_1(monkeypatch):
    monkeypatch.delenv("SOJA_PR_SOWING_MONTH", raising=False)
    monkeypatch.delenv("SOJA_PR_SOWING_DAY", raising=False)
    info = map_info_soja_pr()
    assert (info['sowing_month'], info['sowing_day']) == (11, 1)


@pytest.mark.parametrize("month,day", [(10, 15), (11, 1), (11, 15), (12, 1)])
def test_agro_start_follows_sowing_env(tmp_path, monkeypatch, month, day):
    monkeypatch.setenv("SOJA_PR_SOWING_MONTH", str(month))
    monkeypatch.setenv("SOJA_PR_SOWING_DAY", str(day))
    agro = tmp_path / "a.agro"
    shutil.copy(AGRO, agro)
    update_agro_management_file_soja_pr(str(agro), datetime.datetime(2019, 11, 1))
    cal = _calendar(str(agro))
    assert cal['crop_start_date'] == datetime.date(2019, month, day)
