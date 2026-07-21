import json
import os

import pandas as pd
import requests
from pcse.input import NASAPowerWeatherDataProvider


def setup_paths():
    """Configura e retorna todos os paths necessários para a execução"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agro_dir = os.path.join(base_dir, "inputs", "data", 'agro')

    paths = {
        'BASE': base_dir,
        'DATA': os.path.join(base_dir, "inputs", "data"),
        'COMPLETO': os.path.join(base_dir, "inputs", "data", "completo"),  # Novo caminho
        'AGRO': os.path.join(agro_dir, 'agro_maize.agro'),
        'AGRO_TEMP': os.path.join(agro_dir, 'agro_maize_temp.agro'),
        'CROP': os.path.join(base_dir, "inputs", "data", 'crop'),
        'SOIL': os.path.join(base_dir, "inputs", "data", 'soil', "ec4.new"),
        'YIELD': os.path.join(base_dir, 'inputs', 'data', 'data.csv'),
        'WEATHER_CACHE': os.path.join(base_dir, "inputs","data","weather","weather_cache","weather_cache"),
        'RESULTS': os.path.join(base_dir, "output", "Sensitivity Analysis"),
        'OPTIMIZATION': os.path.join(base_dir, "output", "Optimization")  # Novo caminho para resultados
    }

    # Criar diretórios necessários
    os.makedirs(paths['WEATHER_CACHE'], exist_ok=True)
    os.makedirs(paths['RESULTS'], exist_ok=True)
    os.makedirs(paths['OPTIMIZATION'], exist_ok=True)

    return paths

def map_info():
    """
    Mapeia informações específicas para diferentes países.
    :return: Dicionário onde as chaves são os códigos dos países e os valores são
             dicionários com informações de calendário, duração máxima e arquivo de solo.
    """
    countries_data = {
        'United States': {'calendar': "Apr-Dec", 'max_duration': 244, "soil_file": 'ec6.new'},
        'China':  {'calendar': "Apr-Nov", 'max_duration': 213, "soil_file": 'ec6.new'},
        'Brazil':  {'calendar': "Jan-Oct", 'max_duration': 273, "soil_file": 'ec3.soil'},
        'France':  {'calendar': "Mar-Jan", 'max_duration': 306, "soil_file": 'ec2.soil'},
        'Romania':  {'calendar': "Mar-Jan", 'max_duration': 306, "soil_file": 'ec2.soil'},
        'Italy':  {'calendar': "Mar-Jan", 'max_duration': 306, "soil_file": 'ec2.soil'},
        'Poland':  {'calendar': "Mar-Jan", 'max_duration': 306, "soil_file": 'ec2.soil'},
        'Hungary':  {'calendar': "Mar-Jan", 'max_duration': 306, "soil_file": 'ec2.soil'},
        'Argentina':  {'calendar': "Sep-Aug", 'max_duration': 334, "soil_file": 'ec6.new'},
        'India':  {'calendar': "Mar-Jan", 'max_duration': 306, "soil_file": 'ec4.new'},
        'Mexico':  {'calendar': "Apr-Mar", 'max_duration': 334, "soil_file": 'ec4.new'}
    }
    return countries_data

def get_weather_data_with_cache(latitude, longitude, cache_dir):
    """
    Obtém data climáticos com sistema de cache aprimorado usando arquivos JSON locais

    Args:
        latitude: Latitude do local
        longitude: Longitude do local
        cache_dir: Diretório para armazenar os arquivos de cache

    Returns:
        Provedor de data meteorológicos
    """
    import pickle

    filename = f"weather_lat_{latitude:.4f}_lon_{longitude:.4f}.pkl"
    filepath = os.path.join(cache_dir, filename)

    if os.path.exists(filepath):
        try:
            print(f"Carregando data meteorológicos do cache: {filepath}")
            with open(filepath, 'rb') as f:
                weather_data = pickle.load(f)
            return weather_data
        except Exception as e:
            print(f"Erro ao carregar cache ({e}), obtendo novos data...")

    print(f"Obtendo data meteorológicos para lat={latitude:.4f}, lon={longitude:.4f}")
    try:
        weather_data = NASAPowerWeatherDataProvider(
            latitude=latitude,
            longitude=longitude,
            ETmodel="PM",
            force_update=False
        )

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(weather_data, f)
            print(f"Dados meteorológicos salvos em {filepath}")
        except Exception as e:
            print(f"Erro ao salvar cache: {e}")

        return weather_data

    except Exception as e:
        print(f"Erro na API ({e}), tentando com force_update...")

        weather_data = NASAPowerWeatherDataProvider(
            latitude=latitude,
            longitude=longitude,
            ETmodel="PM",
            force_update=True
        )

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(weather_data, f)
        except Exception as cache_error:
            print(f"Erro ao salvar cache: {cache_error}")

        return weather_data

def update_agro_management_file(agro_path, start_date, country):
    """
    Updates the agromanagement file with the correct start date and duration.

    Args:
        agro_path: Path to the agromanagement YAML file
        start_date: The starting date (datetime object or date string)
        country: Country name to get calendar and duration info
    """
    import yaml
    from datetime import timedelta, datetime

    # Obter informações do país
    countries_data = map_info()
    if country not in countries_data:
        raise ValueError(f"País '{country}' não encontrado em map_info()")

    country_info = countries_data[country]
    calendar = country_info['calendar']
    duration_days = country_info['max_duration']

    # Extrair mês de início do calendário (ex: "Apr-Nov" -> "Apr" -> 4)
    month_abbr = calendar.split('-')[0]
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    start_month = month_map[month_abbr]

    # Converter start_date para datetime
    if not isinstance(start_date, datetime):
        if hasattr(start_date, 'year'):
            crop_start_dt = datetime.combine(start_date, datetime.min.time())
        else:
            crop_start_dt = datetime.fromisoformat(str(start_date))
    else:
        crop_start_dt = start_date

    # Ajustar para o mês correto do calendário
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

def configure_pcse_logging(disable=True):
    """
    Configura o logging do PCSE para evitar conflitos em processamento paralelo

    Args:
        disable: Se True, desativa o logging do PCSE
    """
    import logging
    logger = logging.getLogger('pcse')

    if disable:
        # Desativa completamente o logging do PCSE
        logger.setLevel(logging.CRITICAL)
        # Remove todos os handlers para evitar qualquer escrita em arquivo
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Adiciona um handler null
        logger.addHandler(logging.NullHandler())
    else:
        # Configuração padrão com nível de erro
        logger.setLevel(logging.ERROR)

def WOFOST_bounds(param_list):
    """
    Define os limites dos parâmetros do WOFOST para otimização

    Args:
        param_list: Lista de parâmetros a serem otimizados

    Returns:
        Dicionário com os limites dos parâmetros
    """
    bounds = {
        # Parâmetros de tabelas - AMAXTB
        'AMAXTB000': (1.00, 70.00),  # Taxa máxima de assimilação de CO₂ em DVS 0.00 [kg.ha⁻¹.hr⁻¹]
        'AMAXTB014': (1.00, 70.00),  # Taxa máxima de assimilação de CO₂ em DVS 0.14 [kg.ha⁻¹.hr⁻¹]
        'AMAXTB082': (1.00, 70.00),  # Taxa máxima de assimilação de CO₂ em DVS 0.82 [kg.ha⁻¹.hr⁻¹]
        'AMAXTB200': (1.00, 70.00),  # Taxa máxima de assimilação de CO₂ em DVS 2.00 [kg.ha⁻¹.hr⁻¹]

        # Parâmetros de tabelas - TMPFTB
        'TMPFTB000': (0.000, 1.000),   # Fator de redução do AMAX em temperatura 0.0 °C
        'TMPFTB008': (0.000, 1.000),   # Fator de redução do AMAX em temperatura 8.0 °C
        'TMPFTB020': (0.000, 1.000),   # Fator de redução do AMAX em temperatura 20.0 °C
        'TMPFTB035': (0.000, 1.000),   # Fator de redução do AMAX em temperatura 35.0 °C
        'TMPFTB045': (0.000, 1.000),   # Fator de redução do AMAX em temperatura 45.0 °C

        # Parâmetros de tabelas - EFFTB
        'EFFTB000': (0.400, 0.500),    # Eficiência de uso da luz por folha única em temperatura 0.0 °C [kg.ha⁻¹.hr⁻¹.J ⁻¹.m².s]
        'EFFTB040': (0.400, 0.500),    # Eficiência de uso da luz por folha única em temperatura 40,0 °C [kg.ha⁻¹.hr⁻¹.J ⁻¹.m².s]

        # Parâmetros de tabelas - SLATB
        'SLATB000': (0.0007, 0.0042),  # Área foliar específica em DVS 0.00 [ha.kg⁻¹]
        'SLATB021': (0.0007, 0.0042),  # Área foliar específica em DVS 0.21 [ha.kg⁻¹]
        'SLATB029': (0.0007, 0.0042),  # Área foliar específica em DVS 0.29 [ha.kg⁻¹]
        'SLATB064': (0.0007, 0.0042),  # Área foliar específica em DVS 0.64 [ha.kg⁻¹]
        'SLATB100': (0.0007, 0.0042),  # Área foliar específica em DVS 1.00 [ha.kg⁻¹]
        'SLATB200': (0.0007, 0.0042),  # Área foliar específica em DVS 2.00 [ha.kg⁻¹]

        # Parâmetros de tabelas - RFSETB
        'RFSETB000': (0.250, 1.000),   # Fator de redução da senescência em DVS 0.00
        'RFSETB200': (0.250, 1.000),   # Fator de redução da senescência em DVS 2.00

        # Parâmetros básicos
        'CFET':  (0.8, 1.2),           # Fator de correção da taxa de transpiração
        'CVL':   (0.6, 0.76),          # Eficiência de conversão de assimilados em folhas [kg.kg⁻¹]
        'CVO':   (0.45, 0.85),         # Eficiência de conversão de assimilados em órgãos de armazenamento [kg.kg⁻¹]
        'CVR':   (0.65, 0.76),         # Eficiência de conversão de assimilados em raízes [kg.kg⁻¹]
        'CVS':   (0.63, 0.76),         # Eficiência de conversão de assimilados em caules [kg.kg⁻¹]
        'DEPNR': (1.0, 5.0),           # Número do grupo de culturas para depleção de água do solo
        'DLC':   (6.0, 18.0),          # Fotoperíodo crítico (limite inferior) [hr]
        'DLO':   (6.0, 18.0),          # Fotoperíodo ótimo para desenvolvimento [hr]
        'DVSEND':(1.0, 2.5),           # Estágio de desenvolvimento na colheita
        'DVSI':  (-0.1, 0.5),          # Estágio de desenvolvimento inicial
        'IAIRDU':(0, 1),               # Presença de dutos de ar nas raízes (0=não, 1=sim)
        'IDSL':  (0, 1),               # Dependência do desenvolvimento (0=temperatura, 1=+fotoperíodo, 2=+vernalização)
        'IOX':   (0, 1),               # Efeito de estresse por oxigênio (0=desativado, 1=ativado)
        'PERDL': (0, 0.1),             # Taxa máxima de morte das folhas por estresse hídrico [d⁻¹]
        'Q10':   (1.5, 2),             # Fator Q10 para respiração (aumento por 10 °C)
        'RDMCR': (50.0, 400.0),        # Profundidade máxima de enraizamento [cm]
        'RDI':   (10.0, 50.0),         # Profundidade inicial de enraizamento [cm]
        'RGRLAI':(0.007, 0.5),         # Taxa máxima de crescimento relativo do IAF [d⁻¹]
        'RML':   (0.002, 0.030),       # Taxa relativa de respiração de manutenção das folhas [d⁻¹]
        'RMO':   (0.002, 0.030),       # Taxa relativa de respiração de manutenção dos órgãos de armazenamento [d⁻¹]
        'RMR':   (0.002, 0.030),       # Taxa relativa de respiração de manutenção das raízes [d⁻¹]
        'RMS':   (0.002, 0.030),       # Taxa relativa de respiração de manutenção dos caules [d⁻¹]
        'RRI':   (0.000, 3.0),         # Taxa máxima de crescimento diário das raízes [cm.d⁻¹]
        'SLATB': (0.0007, 0.0042),     # Área foliar específica [ha.kg⁻¹]'SPA': (0.0, 0.0001),
        'SPAN':  (17, 50),             # Duração de vida das folhas a 35 °C [d]
        'TBASE': (-10.0, 10.0),        # Temperatura base para envelhecimento das folhas [°C]
        'TBASEM':(-10.0, 8.0),         # Temperatura base para emergência [°C]
        'TDWI':  (0.5, 300),           # Peso seco inicial total da cultura [kg.ha⁻¹]
        'TEFFMX':(18.0, 32.0),         # Temperatura efetiva máxima para emergência [°C]
        'TSUM1': (150, 1050),          # Soma térmica da emergência até antese [°C.d]
        'TSUM2': (600, 1550),          # Soma térmica da antese até maturidade [°C.d]
        'TSUMEM':(100, 170),           # Soma térmica da semeadura até emergência [°C.d]
    }

    if param_list == "all":
        return bounds

    return {param: bounds[param] for param in param_list if param in bounds}

def _optimization_worker(args):
    """Função wrapper para executar a otimização em paralelo."""
    optimizer_instance, local, ano = args
    return optimizer_instance.optimize_location_year(local, ano)

def get_status_message(result_code):
    """Retorna mensagem de posição da otimização."""
    messages = {
        1: "Sucesso - tolerância relativa atingida",
        2: "Sucesso - tolerância absoluta atingida",
        3: "Sucesso - valor objetivo atingido",
        4: "Sucesso - precisão atingida",
        5: "Máximo de avaliações atingido",
        6: "Tempo máximo excedido",
        -1: "Falha",
        -2: "Valor inválido",
        -3: "Erro de arredondamento",
        -4: "Terminação forçada",
        -5: "Erro desconhecido"
    }
    return messages.get(result_code, "Status desconhecido")

def process_weather_data(weather_file_path):
        """
        Automatiza a leitura e formatação dos data meteorológicos da NASA POWER

        Args:
            weather_file_path (str): Caminho para o arquivo JSON

        Returns:
            pd.DataFrame: DataFrame com colunas date, TS, PS, longitude, latitude
        """
        # Carregar data JSON
        with open(weather_file_path, 'r') as f:
            json_data = json.load(f)

        # Normalizar estrutura JSON
        df_temp = pd.json_normalize(json_data)

        # Extrair data de temperatura (TS) e pressão (PS)
        weather_data = []

        # Obter todas as datas disponíveis
        dates = set()
        for col in df_temp.columns:
            if 'properties.parameter.TS.' in col or 'properties.parameter.PS.' in col:
                date_str = col.split('.')[-1]
                dates.add(date_str)

        # Para cada data, extrair TS e PS
        for date_str in sorted(dates):
            date = pd.to_datetime(date_str, format='%Y%m%d')

            # Buscar colunas de TS e PS para esta data
            ts_col = f'properties.parameter.TS.{date_str}'
            ps_col = f'properties.parameter.PS.{date_str}'

            ts_value = df_temp[ts_col].iloc[0] if ts_col in df_temp.columns else None
            ps_value = df_temp[ps_col].iloc[0] if ps_col in df_temp.columns else None

            weather_data.append({
                'date': date,
                'TS': ts_value,
                'PS': ps_value
            })

        # Criar DataFrame com data meteorológicos
        df_weather = pd.DataFrame(weather_data)

        # Extrair coordenadas geográficas
        coordinates = df_temp['geometry.coordinates'].iloc[0]
        longitude = coordinates[0]
        latitude = coordinates[1]

        # Adicionar coordenadas ao DataFrame
        df_weather['Longitude'] = longitude
        df_weather['Latitude'] = latitude

        return df_weather

def process_all_weather_files(weather_files_list):
    """
    Processa múltiplos arquivos meteorológicos e combina em um DataFrame único

    Args:
        weather_files_list (list): Lista de caminhos para arquivos JSON

    Returns:
        pd.DataFrame: DataFrame combinado com data de todos os arquivos
    """
    all_data = []

    for file_path in weather_files_list:
        df_weather = process_weather_data(file_path)
        all_data.append(df_weather)

    # Combinar todos os DataFrames
    df_combined = pd.concat(all_data, ignore_index=True)

    return df_combined


def obter_elevacao_multiplas_fontes(lat, lon):
    """Tenta obter elevação de múltiplas fontes"""

    # Tentar NASA POWER primeiro
    try:
        api_url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?start=20230101&end=20230101&latitude="
                   f"{lat}&longitude={lon}&community=ag&parameters=WS2M&header=true")
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'geometry' in data and 'coordinates' in data['geometry']:
            return float(data['geometry']['coordinates'][-1]), 'NASA_POWER'
    except:
        pass

    return None, None
