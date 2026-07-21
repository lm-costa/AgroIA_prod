"""
Utilitarios especificos do pipeline paralelo "soja no Parana" (BR-DWGD + IBGE).

Este modulo NAO altera util/utils.py (pipeline de milho global) -- ele reusa
as funcoes genericas de la (WOFOST_bounds, get_status_message,
obter_elevacao_multiplas_fontes, configure_pcse_logging) e define apenas o
que e especifico desta demanda: caminhos proprios (para nao colidir com os
dados de milho), o calendario agricola da soja no PR e uma variante da
atualizacao do arquivo de agromanagement que usa esse calendario.
"""
import os
from datetime import datetime, timedelta

import yaml


def setup_paths_soja_pr():
    """
    Configura os caminhos usados pelo pipeline soja/PR, em uma sub-arvore
    separada de inputs/data e output para nao colidir com o pipeline de
    milho existente.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "inputs", "data", "soja_pr")
    agro_dir = os.path.join(base_dir, "inputs", "data", "agro")

    paths = {
        'BASE': base_dir,
        'DATA': data_dir,
        'COORDINATES': os.path.join(data_dir, "coordinates_pr.xlsx"),
        'COMPLETO': os.path.join(data_dir, "completo"),
        'AGRO': os.path.join(agro_dir, 'agro_soybean_pr.agro'),
        'CROP': os.path.join(base_dir, "inputs", "data", 'crop'),
        'SOIL': os.path.join(base_dir, "inputs", "data", "soil", "ec3.soil"),
        'WEATHER_RAW': r"D:\_py\Clima_AgroIA\data-raw\xavier-data",
        'RESULTS': os.path.join(base_dir, "output", "soja_pr", "Sensitivity Analysis"),
        'OPTIMIZATION': os.path.join(base_dir, "output", "soja_pr", "Optimization"),
    }

    os.makedirs(paths['DATA'], exist_ok=True)
    os.makedirs(paths['COMPLETO'], exist_ok=True)
    os.makedirs(paths['RESULTS'], exist_ok=True)
    os.makedirs(paths['OPTIMIZATION'], exist_ok=True)

    return paths


def map_info_soja_pr():
    """
    Calendario agricola e solo de referencia para a soja no Parana.

    Semeadura em outubro (janela tipica de meados de setembro a novembro no
    PR) e duracao maxima de 200 dias -- e apenas um teto de seguranca para o
    AgroManagement: a colheita real e determinada dinamicamente pelo WOFOST
    quando o estagio de desenvolvimento (DVS) atinge DVSEND, entao ciclos
    mais curtos (grupos de maturacao precoces, ~100-140 dias) terminam antes
    desse teto sem problema.

    O arquivo de solo 'ec3.soil' e o mesmo usado como padrao para o Brasil no
    pipeline de milho (util.utils.map_info); mantido por consistencia -- vale
    revisar/trocar por um perfil especifico dos solos do PR se disponivel.
    """
    return {
        'calendar': "Oct-Mai",
        'sowing_month': 10,
        'max_duration': 200,
        'soil_file': 'ec3.soil',
    }


def update_agro_management_file_soja_pr(agro_path, start_date):
    """
    Variante de util.utils.update_agro_management_file usando o calendario
    da soja no PR (map_info_soja_pr) em vez do map_info() generico do
    pipeline de milho.
    """
    calendar_info = map_info_soja_pr()
    duration_days = calendar_info['max_duration']
    start_month = calendar_info['sowing_month']

    if not isinstance(start_date, datetime):
        if hasattr(start_date, 'year'):
            crop_start_dt = datetime.combine(start_date, datetime.min.time())
        else:
            crop_start_dt = datetime.fromisoformat(str(start_date))
    else:
        crop_start_dt = start_date

    crop_start_dt = crop_start_dt.replace(month=start_month, day=1)
    crop_end_dt = crop_start_dt + timedelta(days=duration_days)

    with open(agro_path, 'r') as f:
        agro_data = yaml.safe_load(f)

    campaign_list = agro_data.get('AgroManagement')
    if not isinstance(campaign_list, list):
        print(f"Warning: 'AgroManagement' key not found or is not a list in {agro_path}")
        return

    if campaign_list:
        campaign_definition = campaign_list[0]
        campaign_details = next(iter(campaign_definition.values()))

        if 'CropCalendar' in campaign_details:
            crop_calendar = campaign_details['CropCalendar']
            crop_calendar['crop_start_date'] = crop_start_dt.date()
            crop_calendar['crop_end_date'] = crop_end_dt.date()

        new_campaign_definition = {crop_start_dt.date(): campaign_details}
        agro_data['AgroManagement'][0] = new_campaign_definition

    with open(agro_path, 'w') as f:
        yaml.dump(agro_data, f, default_flow_style=False, sort_keys=False)


def safra_ano_colheita(date):
    """
    Converte uma data para o 'ano-safra' (ano de colheita) da soja no PR,
    assumindo semeadura em out-nov e colheita em fev-mai do ano seguinte:
    meses >= 7 (jul-dez) pertencem a safra colhida no ano seguinte,
    meses < 7 (jan-jun) pertencem a safra colhida no proprio ano.

    Essa e a mesma convencao usada pelo IBGE/LSPA para rotular a produtividade
    de lavouras de segunda quinzena do ano (a "safra 2015/16" e publicada sob
    ano=2016).
    """
    year = date.year
    month = date.month
    return year + 1 if month >= 7 else year
