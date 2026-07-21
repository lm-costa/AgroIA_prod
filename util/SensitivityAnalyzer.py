import glob
import os
import shutil
import time

import numpy as np
import pandas as pd
import xarray as xr

# Importações do SALib
from SALib.analyze import morris as morris_analyzer_salib
from SALib.sample import morris as morris_sampler

# Importações do PCSE
from pcse.base import ParameterProvider
from pcse.input import CABOFileReader, YAMLAgroManagementReader
from pcse.input import YAMLCropDataProvider
from pcse.input.sitedataproviders import WOFOST72SiteDataProvider
from pcse.util import vap_from_relhum

# Importando funções utilitárias
from utils import WOFOST_bounds, setup_paths, update_agro_management_file

# Define os limites (bounds) dos parâmetros a serem analisados.
DEFAULT_BOUNDS = WOFOST_bounds("all")


class NetCDFDataLoader:
    """
    Classe para carregar e processar dados dos arquivos NetCDF.
    """

    def __init__(self, data_dir, results_dir=None):
            """
            Inicializa o carregador de dados NetCDF.

            Args:
                data_dir: Diretório contendo os arquivos .nc
                results_dir: Diretório onde os resultados são salvos (opcional)
            """
            self.data_dir = data_dir
            self.results_dir = results_dir
            self.nc_files = glob.glob(os.path.join(data_dir, "point_id_*.nc"))
            print(f"Encontrados {len(self.nc_files)} arquivos NetCDF")

    @staticmethod
    def convert_to_wofost_format(ds):
        """
        Converte as variáveis do NetCDF para o formato esperado pelo WOFOST.

        Args:
            ds: Dataset xarray com os dados climáticos

        Returns:
            DataFrame com as variáveis convertidas
        """
        # Converter radiação com tratamento de valores ausentes
        irrad = np.where(
            pd.isna(ds['ALLSKY_SFC_SW_DWN'].values),
            np.nan,
            ds['ALLSKY_SFC_SW_DWN'].values * 1000000.0  # MJ/m²/d para J/m²/d
        )

        # Converter precipitação com tratamento de valores ausentes
        rain = np.where(
            pd.isna(ds['PRECTOTCORR'].values),
            np.nan,
            ds['PRECTOTCORR'].values / 10.0  # mm/d para cm/d
        )

        # Temperatura com tratamento de valores ausentes
        tmin = np.where(pd.isna(ds['T2M_MIN'].values), np.nan, ds['T2M_MIN'].values)
        tmax = np.where(pd.isna(ds['T2M_MAX'].values), np.nan, ds['T2M_MAX'].values)

        # Calcular temperatura média
        tmean = np.where(
            pd.isna(tmin) | pd.isna(tmax),
            np.nan,
            (tmin + tmax) / 2.0
        )

        # Obter umidade relativa
        rh = np.where(pd.isna(ds['RH2M'].values), np.nan, ds['RH2M'].values)  # %

        # Calcular VAP usando o metodo PCSE (retorna kPa, converter para hPa)
        vap = np.array([
            vap_from_relhum(rh_val, temp_val) * 10.0  # kPa para hPa
            if not (pd.isna(rh_val) or pd.isna(temp_val)) else np.nan
            for rh_val, temp_val in zip(rh, tmean)
        ])

        # Velocidade do vento com tratamento de valores ausentes
        wind = np.where(pd.isna(ds['WS2M'].values), np.nan, ds['WS2M'].values)  # m/s

        # Yield e dyield com tratamento de valores ausentes
        yield_values = ds['yield'].values if 'yield' in ds.data_vars else np.full(len(ds['date']), np.nan)
        dyield_values = ds['dyield'].values if 'dyield' in ds.data_vars else np.full(len(ds['date']), np.nan)

        # Substituir valores None ou inválidos por np.nan
        yield_values = np.where(pd.isna(yield_values), np.nan, yield_values)  # kg/ha
        dyield_values = np.where(pd.isna(dyield_values), np.nan, dyield_values)  # kg/ha

        # Criar DataFrame
        df = pd.DataFrame({
            'date': ds['date'].values,
            'IRRAD': irrad,  # J/m²/d
            'TMIN': tmin,  # °C
            'TMAX': tmax,  # °C
            'T2M': tmean,  # °C
            'VAP': vap,  # hPa
            'RAIN': rain,  # cm/d
            'WIND': wind,  # m/s
            'yield': yield_values,  # kg/ha
            'dyield': dyield_values  # kg/ha
        })

        return df

    @staticmethod
    def get_point_info(nc_file):
        """
        Extrai informações do ponto (coordenadas, cluster, ID).

        Args:
            nc_file: Caminho do arquivo NetCDF

        Returns:
            Dicionário com as informações do ponto
        """
        with xr.open_dataset(nc_file) as ds:
            point_id = os.path.basename(nc_file).replace('point_id_', '').replace('.nc', '')

            info = {
                'point_id': point_id,
                'latitude': float(ds.attrs['latitude']),
                'longitude': float(ds.attrs['longitude']),
                'cluster_id': ds.attrs.get('cluster_id', None),
                'country': ds.attrs.get('country', 'Brazil'),  # Extrair país do arquivo
                'beta_0': ds.variables['dyield'].attrs['beta_0'],
                'beta_1': ds.variables['dyield'].attrs['beta_1']
            }

        return info

    def load_point_data(self, nc_file):
        """
        Carrega todos os dados de um ponto específico.

        Args:
            nc_file: Caminho do arquivo NetCDF

        Returns:
            Tupla (info_dict, dataframe_convertido)
        """
        with xr.open_dataset(nc_file) as ds:
            info = self.get_point_info(nc_file)

            try:
                if 'elevation_m' in ds.attrs:
                    elev_value = ds.attrs['elevation_m']
                    if elev_value is None or pd.isna(elev_value):
                        info['elevation'] = np.nan
                    else:
                        info['elevation'] = float(elev_value)
                else:
                    info['elevation'] = np.nan
            except (ValueError, TypeError, KeyError):
                info['elevation'] = np.nan

            df = self.convert_to_wofost_format(ds)

        return info, df

    def sample_points_by_cluster(self, n_points_per_cluster=100):
                """
                Seleciona amostra estratificada de pontos por cluster, considerando análises já realizadas.

                Args:
                    n_points_per_cluster: Número total de pontos desejados por cluster

                Returns:
                    Lista de caminhos dos arquivos selecionados
                """
                # Verificar arquivos já analisados (apenas se results_dir foi fornecido)
                analyzed_by_cluster = {}

                if self.results_dir and os.path.exists(self.results_dir):
                    existing_results = glob.glob(os.path.join(self.results_dir, "SA_MORRIS_cluster*_point*.csv"))

                    # Extrair pontos já analisados por cluster
                    for result_file in existing_results:
                        basename = os.path.basename(result_file)
                        # Formato: SA_MORRIS_cluster{cluster_id}_point{point_id}.csv
                        if basename.startswith("SA_MORRIS_cluster") and "_point" in basename:
                            try:
                                cluster_part = basename.split("_cluster")[1].split("_point")[0]
                                point_part = basename.split("_point")[1].replace(".csv", "")
                                cluster_id = int(cluster_part)
                                point_id = point_part

                                if cluster_id not in analyzed_by_cluster:
                                    analyzed_by_cluster[cluster_id] = []
                                analyzed_by_cluster[cluster_id].append(point_id)
                            except (IndexError, ValueError):
                                continue

                # Obter cluster_id de todos os arquivos
                cluster_mapping = {}

                for nc_file in self.nc_files:
                    info = self.get_point_info(nc_file)
                    cluster_id = info['cluster_id']
                    point_id = info['point_id']

                    if cluster_id not in cluster_mapping:
                        cluster_mapping[cluster_id] = []

                    cluster_mapping[cluster_id].append((nc_file, point_id))

                print(f"\n{'='*60}")
                print("VERIFICAÇÃO DE PONTOS ANALISADOS")
                print(f"{'='*60}")

                for cluster_id in sorted(cluster_mapping.keys()):
                    n_analyzed = len(analyzed_by_cluster.get(cluster_id, []))
                    n_total = len(cluster_mapping[cluster_id])
                    print(f"Cluster {cluster_id}: {n_analyzed}/{n_points_per_cluster} pontos já analisados (total disponível: {n_total})")

                # Amostrar pontos de cada cluster
                selected_files = []

                for cluster_id, files_info in cluster_mapping.items():
                    analyzed_points = set(analyzed_by_cluster.get(cluster_id, []))
                    n_already_analyzed = len(analyzed_points)

                    # Calcular quantos pontos ainda precisam ser analisados
                    n_remaining = n_points_per_cluster - n_already_analyzed

                    if n_remaining <= 0:
                        print(f"\n✅ Cluster {cluster_id}: Meta de {n_points_per_cluster} pontos já atingida!")
                        continue

                    # Filtrar arquivos não analisados
                    available_files = [
                        (nc_file, point_id) for nc_file, point_id in files_info
                        if point_id not in analyzed_points
                    ]

                    if not available_files:
                        print(f"\n⚠️  Cluster {cluster_id}: Sem novos pontos disponíveis para análise")
                        continue

                    # Amostrar apenas os pontos necessários
                    n_sample = min(n_remaining, len(available_files))

                    # Extrair apenas os caminhos dos arquivos para amostragem
                    available_nc_files = [nc_file for nc_file, _ in available_files]
                    sampled = np.random.choice(available_nc_files, size=n_sample, replace=False)
                    selected_files.extend(sampled)

                    print(f"\n📊 Cluster {cluster_id}:")
                    print(f"   - Já analisados: {n_already_analyzed}")
                    print(f"   - Restantes necessários: {n_remaining}")
                    print(f"   - Disponíveis para análise: {len(available_files)}")
                    print(f"   - Selecionados agora: {n_sample}")

                print(f"\n{'='*60}")
                print(f"TOTAL DE NOVOS PONTOS SELECIONADOS: {len(selected_files)}")
                print(f"{'='*60}\n")

                return selected_files


class MorrisScreeningAnalyzer:
            """
            Classe dedicada a realizar a análise de sensibilidade (screening)
            pelo metodo de Morris para o modelo WOFOST usando dados NetCDF.
            """

            def __init__(self, bounds, paths, nc_loader):
                """
                Inicializa o analisador de sensibilidade.

                Args:
                    bounds: Dicionário com os limites dos parâmetros
                    paths: Dicionário com os caminhos dos arquivos
                    nc_loader: Instância de NetCDFDataLoader
                """
                self.bounds = bounds
                self.paths = paths
                self.nc_loader = nc_loader
                self.problem = {
                    'num_vars': len(bounds),
                    'names': list(bounds.keys()),
                    'bounds': list(bounds.values())
                }
                import logging
                self.logger = logging.getLogger('pcse')
                self.logger.setLevel(logging.ERROR)

            def extract_model_params(self, X):
                """
                Extrai parâmetros do modelo a partir do vetor de entrada X.

                Args:
                    X: Vetor com valores dos parâmetros

                Returns:
                    Dicionário com parâmetros para o modelo
                """
                model_params = {}
                for i, param_name in enumerate(self.problem['names']):
                    model_params[param_name] = X[i]

                return model_params

            def run_wofost_model(self, model_params, parameters, weather, agromanagement):
                """
                Executa o modelo WOFOST com parâmetros especificados.

                Args:
                    model_params: Dicionário com parâmetros do modelo
                    parameters: ParameterProvider do WOFOST
                    weather: WeatherDataProvider
                    agromanagement: AgroManagement

                Returns:
                    Produtividade simulada (kg/ha)
                """
                from pcse.models import Wofost72_WLP_FD

                try:
                    # Configurar parâmetros SLATB (tabela específica de área foliar)
                    if any(key.startswith('SLATB') for key in model_params):
                        slatb_values = []
                        for dvs, key in [(0.00, 'SLATB000'), (0.21, 'SLATB021'), (0.29, 'SLATB029'),
                                         (0.64, 'SLATB064'), (1.00, 'SLATB100'), (2.00, 'SLATB200')]:
                            if key in model_params:
                                slatb_values.extend([dvs, model_params[key]])

                        if slatb_values:
                            parameters.set_override("SLATB", slatb_values)

                    # Configurar parâmetros AMAXTB (tabela de fotossíntese máxima)
                    if any(key.startswith('AMAXTB') for key in model_params):
                        amaxtb_values = []
                        for dvs, key in [(0.00, 'AMAXTB000'), (0.14, 'AMAXTB014'),
                                         (0.82, 'AMAXTB082'), (2.00, 'AMAXTB200')]:
                            if key in model_params:
                                amaxtb_values.extend([dvs, model_params[key]])

                        if amaxtb_values:
                            parameters.set_override('AMAXTB', amaxtb_values)

                    # Configurar parâmetros TMPFTB (tabela de redução por temperatura)
                    if any(key.startswith('TMPFTB') for key in model_params):
                        tmpftb_values = []
                        for tmp, key in [(0.00, 'TMPFTB000'), (8.00, 'TMPFTB008'), (20.0, 'TMPFTB020'),
                                         (35.0, 'TMPFTB035'), (45.0, 'TMPFTB045')]:
                            if key in model_params:
                                tmpftb_values.extend([tmp, model_params[key]])

                        if tmpftb_values:
                            parameters.set_override("TMPFTB", tmpftb_values)

                    # Configurar parâmetros RFSETB (tabela de particionamento)
                    if any(key.startswith('RFSETB') for key in model_params):
                        rfsetb_values = []
                        for dvs, key in [(0, 'RFSETB000'), (2, 'RFSETB200')]:
                            if key in model_params:
                                rfsetb_values.extend([dvs, model_params[key]])

                        if rfsetb_values:
                            parameters.set_override("RFSETB", rfsetb_values)

                    # Configurar parâmetros simples (não são tabelas)
                    simple_params = [
                        'SPAN', 'TSUM1', 'TDWI', 'RGRLAI', 'TSUMEM', 'TBASE', 'CVL',
                        'CVR', 'CVS', 'Q10', 'RML', 'RMR', 'RMS', 'RDI', 'RRI', 'TSUM2',
                        'CFET', 'CVO', 'DEPNR', 'DLC', 'DLO', 'DVSEND', 'DVSI', 'IAIRDU',
                        'IDSL', 'IOX', 'PERDL', 'RDMCR', 'REFCO2L', 'RMO', 'SPA', 'TBASEM',
                        'TEFFMX'
                    ]

                    for param in simple_params:
                        if param in model_params:
                            parameters.set_override(param, model_params[param])

                    # Validação: DLO deve ser maior que DLC
                    if 'DLC' in model_params and 'DLO' in model_params:
                        if model_params['DLO'] <= model_params['DLC']:
                            dlc_adjusted = model_params['DLC']
                            dlo_adjusted = dlc_adjusted + 0.1
                            parameters.set_override('DLC', dlc_adjusted)
                            parameters.set_override('DLO', dlo_adjusted)

                    # Inicializar e executar modelo
                    wofost = Wofost72_WLP_FD(parameters, weather, agromanagement)
                    wofost.run_till_terminate()

                    # Obter resultado final
                    output = wofost.get_output()
                    if output:
                        final_output = output[-1]
                        yield_kg_ha = final_output.get('TWSO', 0) * 1000  # t/ha para kg/ha
                        return yield_kg_ha
                    else:
                        return np.nan

                except Exception as e:
                    self.logger.error(f"Erro na simulação WOFOST: {e}")
                    return np.nan

            def create_weather_data_provider(self, weather_df, latitude, longitude, elevation=None):
                    """
                    Cria um provider de dados climáticos compatível com NASA POWER.
                    """
                    from pcse.base import WeatherDataProvider, WeatherDataContainer
                    from pcse.util import penman, penman_monteith

                    class NetCDFWeatherDataProvider(WeatherDataProvider):
                        def __init__(self, weather_data, lat, lon, elev):
                            WeatherDataProvider.__init__(self)

                            self.latitude = lat
                            self.longitude = lon
                            self.elevation = elev if not pd.isna(elev) else np.nan

                            # Coeficientes Angstrom (valores padrão do NASA POWER)
                            self.angstA = 0.29
                            self.angstB = 0.49

                            self.weather_records = {}
                            for _, row in weather_data.iterrows():
                                # Conversão de data
                                date_value = row['date']
                                if isinstance(date_value, pd.Timestamp):
                                    day = date_value.date()
                                elif isinstance(date_value, np.datetime64):
                                    day = pd.Timestamp(date_value).date()
                                else:
                                    day = pd.to_datetime(date_value).date()

                                try:
                                    # Calcular E0 e ES0 usando Penman
                                    E0, ES0, _ = penman(
                                        DAY=day,
                                        LAT=lat,
                                        ELEV=self.elevation,
                                        TMIN=row['TMIN'],
                                        TMAX=row['TMAX'],
                                        AVRAD=row['IRRAD'],
                                        VAP=row['VAP'],
                                        WIND2=row['WIND'],
                                        ANGSTA=self.angstA,
                                        ANGSTB=self.angstB
                                    )

                                    # Calcular ET0 usando Penman-Monteith
                                    ET0 = penman_monteith(
                                        DAY=day,
                                        LAT=lat,
                                        ELEV=self.elevation,
                                        TMIN=row['TMIN'],
                                        TMAX=row['TMAX'],
                                        AVRAD=row['IRRAD'],
                                        VAP=row['VAP'],
                                        WIND2=row['WIND']
                                    )

                                    # Converter de mm/d para cm/d
                                    E0 = E0 / 10.0
                                    ES0 = ES0 / 10.0
                                    ET0 = ET0 / 10.0

                                except Exception as e:
                                    print(f"Aviso: Erro ao calcular ET para {day}: {e}. Usando valores padrão.")
                                    E0, ES0, ET0 = 0.0, 0.0, 0.0

                                wdc = WeatherDataContainer(
                                    DAY=day,
                                    LAT=lat,
                                    LON=lon,
                                    ELEV=self.elevation,
                                    IRRAD=row['IRRAD'],
                                    TMIN=row['TMIN'],
                                    TMAX=row['TMAX'],
                                    VAP=row['VAP'],
                                    RAIN=row['RAIN'],
                                    WIND=row['WIND'],
                                    E0=E0,
                                    ES0=ES0,
                                    ET0=ET0
                                )
                                self.weather_records[day] = wdc

                            self._first_date = min(self.weather_records.keys())
                            self._last_date = max(self.weather_records.keys())

                        def __call__(self, day, *args, **kwargs):
                            return self.weather_records.get(day)

                    if elevation is None or pd.isna(elevation):
                        elevation = np.nan

                    return NetCDFWeatherDataProvider(weather_df, latitude, longitude, elevation)

            def execute_morris_screening_for_point(self, params):
                        """
                        Executa a análise de Morris para um ponto específico.

                        Args:
                            params: Tupla com (nc_file, param_values, cluster_id)

                        Returns:
                            DataFrame com resultados de Morris ou None em caso de erro
                        """
                        nc_file, param_values, cluster_id = params

                        # Extrair info do ponto
                        point_info, weather_df = self.nc_loader.load_point_data(nc_file)
                        point_id = point_info['point_id']

                        print(f"\nIniciando Morris Screen para: Point {point_id} (Cluster {cluster_id})")

                        result_file = os.path.join(
                            self.paths['RESULTS'],
                            f"SA_MORRIS_cluster{cluster_id}_point{point_id}.csv"
                        )

                        if os.path.exists(result_file):
                            print(f"Resultado já existe para Point {point_id}. Carregando...")
                            return pd.read_csv(result_file)

                        try:
                            # Preparar dados para simulação
                            LAT = point_info['latitude']
                            LON = point_info['longitude']

                            # Determinar data de plantio (primeiro dia com dados)
                            start_date = pd.to_datetime(weather_df['date'].iloc[0])

                            # Obter informações do país do dataset
                            from utils import map_info
                            countries_info = map_info()
                            location = point_info.get('country', np.nan)  # Verificar nome correto do atributo
                            duration_days = countries_info.get(location, {}).get('max_duration', np.nan)
                            soil_file = countries_info.get(location, {}).get('soil_file', np.nan)

                            # Carregar dados de cultura, solo e sítio
                            cropfile = YAMLCropDataProvider(fpath=self.paths['CROP'])
                            cropfile.set_active_crop('maize', "Maize_VanHeemst_1988")

                            soil_path = os.path.join(os.path.dirname(self.paths['SOIL']), soil_file)
                            soildata = CABOFileReader(fname=soil_path)
                            sitedata = WOFOST72SiteDataProvider(WAV=100)

                            elevation = point_info.get('elevation', np.nan)
                            weather = self.create_weather_data_provider(weather_df, LAT, LON, elevation)

                            # Atualizar arquivo de agro manejo
                            agro_path_temp = self.paths['AGRO'] + f'_temp_{point_id}.yaml'
                            shutil.copy(self.paths['AGRO'], agro_path_temp)
                            update_agro_management_file(
                                agro_path_temp,
                                start_date,
                                country=location
                            )

                            agromanagement = YAMLAgroManagementReader(agro_path_temp)
                            parameters = ParameterProvider(
                                cropdata=cropfile,
                                soildata=soildata,
                                sitedata=sitedata
                            )

                            # Executar simulações Morris
                            Y = np.zeros(param_values.shape[0])

                            print(f"Executando {param_values.shape[0]} simulações para Point {point_id}...")

                            for i, X in enumerate(param_values):
                                if i % 100 == 0:
                                    print(f"  Progresso: {i}/{param_values.shape[0]} ({100*i/param_values.shape[0]:.1f}%)")

                                model_params = self.extract_model_params(X)
                                Y[i] = self.run_wofost_model(
                                    model_params,
                                    parameters,
                                    weather,
                                    agromanagement
                                )

                            # Analisar resultados
                            Si = morris_analyzer_salib.analyze(
                                self.problem,
                                param_values,
                                Y,
                                print_to_console=False
                            )

                            morris_results = pd.DataFrame({
                                'parameter': Si['names'],
                                'mu_star': Si['mu_star'],
                                'sigma': Si['sigma'],
                                'mu': Si['mu'],
                                'point_id': point_id,
                                'cluster_id': cluster_id,
                                'latitude': LAT,
                                'longitude': LON
                            })

                            # Salvar resultados
                            morris_results.to_csv(result_file, index=False)
                            print(f"Resultados salvos em {result_file}")

                            # Limpar arquivo temporário
                            if os.path.exists(agro_path_temp):
                                os.remove(agro_path_temp)

                            return morris_results

                        except Exception as e:
                            print(f"Erro ao executar screening para Point {point_id}: {e}")
                            import traceback
                            traceback.print_exc()
                            return None

            def run_analysis(self, n_points_per_cluster=100):
                    """
                    Orquestra a análise de sensibilidade usando dados NetCDF.

                    Args:
                        n_points_per_cluster: Número de pontos a amostrar por cluster
                    """
                    print("=== INICIANDO ANÁLISE DE SENSIBILIDADE (MORRIS SCREEN) ===")

                    # Amostrar pontos por cluster (considera arquivos já analisados)
                    selected_files = self.nc_loader.sample_points_by_cluster(n_points_per_cluster)

                    if not selected_files:
                        print("\n✅ Todos os pontos já foram analisados!")
                        return None

                    # Gerar amostras de Morris
                    morris_samples = morris_sampler.sample(self.problem, N=100, num_levels=4)
                    print(f"Amostra de Morris gerada com {morris_samples.shape[0]} pontos.")

                    # Preparar tarefas por cluster
                    tasks = []
                    for nc_file in selected_files:
                        point_info = self.nc_loader.get_point_info(nc_file)
                        cluster_id = point_info['cluster_id']
                        tasks.append((nc_file, morris_samples, cluster_id))

                    # Executar análises com barra de progresso
                    screening_results_list = []
                    total_tasks = len(tasks)
                    successful = 0
                    failed = 0

                    print(f"\n{'='*60}")
                    print(f"EXECUTANDO {total_tasks} ANÁLISES")
                    print(f"{'='*60}\n")

                    for i, task in enumerate(tasks, 1):
                        # Barra de progresso
                        progress = (i / total_tasks) * 100
                        bar_length = 40
                        filled = int(bar_length * i / total_tasks)
                        bar = '█' * filled + '░' * (bar_length - filled)

                        print(f"\n[{bar}] {progress:.1f}% ({i}/{total_tasks})")
                        print(f"{'─'*60}")

                        try:
                            result = self.execute_morris_screening_for_point(task)
                            if result is not None:
                                screening_results_list.append(result)
                                successful += 1
                                print(f"✅ Análise concluída com sucesso!")
                            else:
                                failed += 1
                                print(f"❌ Análise retornou resultado nulo")
                        except Exception as e:
                            failed += 1
                            print(f"❌ Tarefa falhou: {e}")

                    # Resumo final
                    print(f"\n{'='*60}")
                    print(f"RESUMO DA EXECUÇÃO")
                    print(f"{'='*60}")
                    print(f"✅ Sucessos: {successful}/{total_tasks}")
                    print(f"❌ Falhas: {failed}/{total_tasks}")
                    print(f"{'='*60}\n")

                    if not screening_results_list:
                        print("⚠️  Nenhum resultado válido foi obtido.")
                        return None

                    # Analisar resultados agregados
                    important_params = self.analyze_overall_results(screening_results_list)

                    # Análise por cluster
                    self.analyze_results_by_cluster(screening_results_list)

                    return important_params

            def analyze_results_by_cluster(self, screening_results_list):
                """
                Analisa os resultados de sensibilidade separados por cluster.

                Args:
                    screening_results_list: Lista de DataFrames com resultados
                """
                if not screening_results_list:
                    return

                combined = pd.concat(screening_results_list)

                print("\n\n" + "=" * 60)
                print("      ANÁLISE DE SENSIBILIDADE POR CLUSTER")
                print("=" * 60)

                for cluster_id in sorted(combined['cluster_id'].unique()):
                    cluster_data = combined[combined['cluster_id'] == cluster_id]

                    grouped = cluster_data.groupby('parameter').agg(
                        mu_star=('mu_star', 'mean'),
                        sigma=('sigma', 'mean'),
                        mu=('mu', 'mean')
                    ).reset_index()

                    sorted_params = grouped.sort_values(by='mu_star', ascending=False)

                    print(f"\n--- Cluster {cluster_id} ---")
                    print(f"Número de pontos analisados: {cluster_data['point_id'].nunique()}")
                    print("\nTop 10 parâmetros mais sensíveis:")
                    print(sorted_params.head(10))

                    # Salvar resultados agregados por cluster
                    cluster_file = os.path.join(
                        self.paths['RESULTS'],
                        f"SA_MORRIS_cluster{cluster_id}_summary.csv"
                    )
                    sorted_params.to_csv(cluster_file, index=False)
                    print(f"Resumo salvo em {cluster_file}")

                print("=" * 60)

            def analyze_overall_results(self, screening_results_list):
                """
                Analisa os resultados agregados de todos os clusters.

                Args:
                    screening_results_list: Lista de DataFrames com resultados

                Returns:
                    DataFrame com parâmetros importantes
                """
                if not screening_results_list:
                    print("Nenhum resultado para analisar.")
                    return None

                # Combinar todos os resultados
                combined = pd.concat(screening_results_list, ignore_index=True)

                # Agregar por parâmetro
                grouped = combined.groupby('parameter').agg({
                    'mu_star': 'mean',
                    'sigma': 'mean',
                    'mu': 'mean'
                })

                # Ordenar por mu_star (sensibilidade)
                sorted_params = grouped.sort_values('mu_star', ascending=False)

                print("\n\n" + "=" * 60)
                print("      ANÁLISE GERAL DE SENSIBILIDADE")
                print("=" * 60)
                print(f"\nNúmero total de pontos analisados: {combined['point_id'].nunique()}")
                print(f"Número de clusters: {combined['cluster_id'].nunique()}")
                print("\nTop 15 parâmetros mais sensíveis (média geral):")
                print(sorted_params.head(15))

                # Salvar resultados gerais
                overall_file = os.path.join(
                    self.paths['RESULTS'],
                    "SA_MORRIS_overall_summary.csv"
                )
                sorted_params.to_csv(overall_file)
                print(f"\nResumo geral salvo em {overall_file}")

                # Identificar parâmetros importantes (mu_star > threshold)
                threshold = sorted_params['mu_star'].quantile(0.75)  # Top 25%
                important_params = sorted_params[sorted_params['mu_star'] > threshold]

                print(f"\nParâmetros importantes (mu_star > {threshold:.2f}):")
                print(important_params.index.tolist())

                return important_params


def main():
    """Função principal que inicia e finaliza o processo."""
    start_time = time.time()

    # Configurar caminhos
    paths = setup_paths()

    # Criar diretório de resultados se não existir
    os.makedirs(paths['RESULTS'], exist_ok=True)

    # Inicializar carregador de dados NetCDF com diretório de resultados
    data_dir = os.path.join(
        paths['BASE'],
        'inputs',
        'data',
        'completo'
    )

    nc_loader = NetCDFDataLoader(data_dir, results_dir=paths['RESULTS'])

    # Criar analisador
    analyzer = MorrisScreeningAnalyzer(DEFAULT_BOUNDS, paths, nc_loader)

    # Executar análise (100 pontos por cluster)
    results = analyzer.run_analysis(n_points_per_cluster=50)

    if results is not None:
        elapsed_time = time.time() - start_time
        print(f"\n✅ Análise de Morris Screen finalizada com sucesso!")
        print(f"⏱️  Tempo total: {elapsed_time / 3600:.2f} horas")
    else:
        print("\nℹ️  A análise não produziu resultados.")

if __name__ == '__main__':
    main()
