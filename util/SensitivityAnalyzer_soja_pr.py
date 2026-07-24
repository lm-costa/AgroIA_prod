"""
Variante do MorrisScreeningAnalyzer (util/SensitivityAnalyzer.py) para a
cultura da soja no Parana, usando dados climaticos do BR-DWGD.

So sobrescreve o que e especifico de cultura/calendario
(execute_morris_screening_for_point). Todo o resto -- extracao de parametros,
execucao do WOFOST, montagem do provider de clima, orquestracao do
screening -- e reaproveitado sem alteracao de SensitivityAnalyzer.py, para
nao mexer no pipeline de milho existente.
"""
import os
import shutil

import numpy as np
import pandas as pd

from pcse.base import ParameterProvider
from pcse.input import CABOFileReader, YAMLAgroManagementReader
from pcse.input import YAMLCropDataProvider
from pcse.input.sitedataproviders import WOFOST72SiteDataProvider

from SensitivityAnalyzer import MorrisScreeningAnalyzer, NetCDFDataLoader
from utils_soja_pr import map_info_soja_pr, update_agro_management_file_soja_pr

# Importações do SALib usadas apenas dentro do método sobrescrito
from SALib.analyze import morris as morris_analyzer_salib


class SoyNetCDFDataLoader(NetCDFDataLoader):
    """
    NetCDFDataLoader para os arquivos point_id_*.nc gerados pelo pipeline
    soja/PR (3.Climatic_data_Soja_PR.ipynb), que ja armazenam as variaveis no
    formato do WOFOST (IRRAD, TMIN, TMAX, T2M, VAP, RAIN, WIND).

    O NetCDFDataLoader original (pipeline de milho) espera os nomes brutos da
    NASA POWER (ALLSKY_SFC_SW_DWN, PRECTOTCORR, T2M_MAX/MIN, RH2M, WS2M) e
    faz a conversao de unidades dentro de convert_to_wofost_format -- aqui a
    conversao ja foi feita na etapa 3, entao so precisamos remontar o
    DataFrame a partir das variaveis existentes, sem reconverter nada.
    """

    @staticmethod
    def convert_to_wofost_format(ds):
        n = len(ds['date'])
        yield_values = ds['yield'].values if 'yield' in ds.data_vars else np.full(n, np.nan)
        dyield_values = ds['dyield'].values if 'dyield' in ds.data_vars else np.full(n, np.nan)

        return pd.DataFrame({
            'date': ds['date'].values,
            'IRRAD': ds['IRRAD'].values,
            'TMIN': ds['TMIN'].values,
            'TMAX': ds['TMAX'].values,
            'T2M': ds['T2M'].values,
            'VAP': ds['VAP'].values,
            'RAIN': ds['RAIN'].values,
            'WIND': ds['WIND'].values,
            'yield': yield_values,
            'dyield': dyield_values,
        })


class SoyMorrisScreeningAnalyzerPR(MorrisScreeningAnalyzer):
    """Morris screening para soja/PR, reaproveitando a infraestrutura do milho."""

    CROP_NAME = 'soybean'
    VARIETY_NAME = 'Soybean_VanHeemst_1988'

    def execute_morris_screening_for_point(self, params):
        nc_file, param_values, cluster_id = params

        point_info, weather_df = self.nc_loader.load_point_data(nc_file)
        point_id = point_info['point_id']

        print(f"\nIniciando Morris Screen (soja/PR) para: Point {point_id} (Cluster {cluster_id})")

        result_file = os.path.join(
            self.paths['RESULTS'],
            f"SA_MORRIS_cluster{cluster_id}_point{point_id}.csv"
        )

        if os.path.exists(result_file):
            print(f"Resultado ja existe para Point {point_id}. Carregando...")
            return pd.read_csv(result_file)

        try:
            LAT = point_info['latitude']
            LON = point_info['longitude']

            start_date = pd.to_datetime(weather_df['date'].iloc[0])

            calendar_info = map_info_soja_pr()
            soil_file = calendar_info['soil_file']

            cropfile = YAMLCropDataProvider(fpath=self.paths['CROP'])
            cropfile.set_active_crop(self.CROP_NAME, self.VARIETY_NAME)

            soil_path = os.path.join(os.path.dirname(self.paths['SOIL']), soil_file)
            soildata = CABOFileReader(fname=soil_path)
            sitedata = WOFOST72SiteDataProvider(WAV=100)

            elevation = point_info.get('elevation', np.nan)
            weather = self.create_weather_data_provider(weather_df, LAT, LON, elevation)

            agro_path_temp = self.paths['AGRO'] + f'_temp_{point_id}.yaml'
            shutil.copy(self.paths['AGRO'], agro_path_temp)
            update_agro_management_file_soja_pr(agro_path_temp, start_date)

            agromanagement = YAMLAgroManagementReader(agro_path_temp)
            parameters = ParameterProvider(
                cropdata=cropfile,
                soildata=soildata,
                sitedata=sitedata
            )

            Y = np.zeros(param_values.shape[0])

            print(f"Executando {param_values.shape[0]} simulacoes para Point {point_id}...")

            for i, X in enumerate(param_values):
                if i % 100 == 0:
                    print(f"  Progresso: {i}/{param_values.shape[0]} ({100 * i / param_values.shape[0]:.1f}%)")

                model_params = self.extract_model_params(X)
                Y[i] = self.run_wofost_model(
                    model_params,
                    parameters,
                    weather,
                    agromanagement
                )

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

            morris_results.to_csv(result_file, index=False)
            print(f"Resultados salvos em {result_file}")

            if os.path.exists(agro_path_temp):
                os.remove(agro_path_temp)

            return morris_results

        except Exception as e:
            print(f"Erro ao executar screening para Point {point_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
