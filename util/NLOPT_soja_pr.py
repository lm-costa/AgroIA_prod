"""
Variante do WOFOSTMultiYearOptimizer (util/NLOPT_MultiYear.py) para a soja no
Parana, usando dados climaticos do BR-DWGD.

Diferente do pipeline de milho -- onde CLUSTER_PARAMS vem de um Morris
Screen ja rodado e fixado como dicionario no codigo -- aqui o ranking de
parametros por cluster e carregado em tempo de execucao a partir do
resultado da etapa de Sensitivity Analysis (6.SA_Soja_PR.ipynb), ja que essa
analise ainda nao existe para soja/PR. Tudo o que e generico (simulacao do
WOFOST, funcao objetivo multi-anual, calculo de metricas, persistencia dos
resultados) e reaproveitado sem alteracao de NLOPT.py / NLOPT_MultiYear.py.
"""
import os
import shutil

import numpy as np
import pandas as pd

from pcse.base import ParameterProvider
from pcse.input import CABOFileReader, YAMLAgroManagementReader
from pcse.input import YAMLCropDataProvider
from pcse.input.sitedataproviders import WOFOST72SiteDataProvider

from NLOPT_MultiYear import WOFOSTMultiYearOptimizer
from utils_soja_pr import map_info_soja_pr, update_agro_management_file_soja_pr


class SoyWOFOSTMultiYearOptimizerPR(WOFOSTMultiYearOptimizer):
    """Calibracao multi-anual do WOFOST para soja/PR (BR-DWGD + IBGE)."""

    CROP_NAME = 'soybean'
    VARIETY_NAME = 'Soybean_VanHeemst_1988'

    def __init__(self, paths, nc_loader, cluster_params, algorithm=None, max_eval=3000):
        import nlopt
        if algorithm is None:
            algorithm = nlopt.LN_BOBYQA
        super().__init__(paths, nc_loader, algorithm, max_eval)
        # Sobrescreve o CLUSTER_PARAMS herdado (calibrado para milho) pelo
        # ranking especifico de soja/PR obtido na etapa de Sensitivity Analysis.
        self.CLUSTER_PARAMS = cluster_params

    def prepare_multiyear_context(self, point_info, weather_df, cluster_id):
        LAT = point_info['latitude']
        LON = point_info['longitude']
        elevation = point_info.get('elevation', np.nan)

        calendar_info = map_info_soja_pr()
        soil_file = calendar_info['soil_file']

        cropfile = YAMLCropDataProvider(fpath=self.paths['CROP'])
        cropfile.set_active_crop(self.CROP_NAME, self.VARIETY_NAME)

        soil_path = os.path.join(os.path.dirname(self.paths['SOIL']), soil_file)
        soildata = CABOFileReader(fname=soil_path)
        sitedata = WOFOST72SiteDataProvider(WAV=100)

        weather_df['year'] = pd.to_datetime(weather_df['date']).dt.year
        years_with_data = weather_df[weather_df['dyield'].notna()]['year'].unique()

        years_data = []

        for year in sorted(years_with_data):
            year_data_df = weather_df[weather_df['year'] == year].copy()
            dyield_obs = year_data_df['yield'].dropna()

            if len(dyield_obs) == 0:
                continue

            weather = self.create_weather_data_provider(year_data_df, LAT, LON, elevation)

            agro_path_temp = f"{self.paths['AGRO']}_temp_{cluster_id}_{year}.yaml"
            shutil.copy(self.paths['AGRO'], agro_path_temp)
            start_date = pd.to_datetime(year_data_df['date'].iloc[0])
            update_agro_management_file_soja_pr(agro_path_temp, start_date)
            agromanagement = YAMLAgroManagementReader(agro_path_temp)

            parameters = ParameterProvider(cropdata=cropfile, soildata=soildata, sitedata=sitedata)

            years_data.append({
                'year': year,
                'weather': weather,
                'agromanagement': agromanagement,
                'parameters': parameters,
                'dyield_target': np.mean(dyield_obs.values),
                'agro_temp_file': agro_path_temp
            })

        return years_data


def load_cluster_params_from_ranking(ranking_json_path, top_n=44):
    """
    Le o ranking de sensibilidade por cluster salvo por 6.SA_Soja_PR.ipynb e
    monta o dicionario {cluster_id: [param_names ordenados por mu_star]} no
    mesmo formato que WOFOSTOptimizer.CLUSTER_PARAMS usa para o milho.

    Espera um JSON no formato:
        {"<cluster_id>": [{"Rank":1, "parameter": "TSUM1", "mu_star": ..., "sigma": ...}, ...], ...}
    """
    import json

    with open(ranking_json_path, 'r') as f:
        ranking = json.load(f)

    cluster_params = {}
    for cluster_id_str, params_ranked in ranking.items():
        cluster_id = float(cluster_id_str)
        names = [p['parameter'] for p in params_ranked[:top_n]]
        cluster_params[cluster_id] = names

    return cluster_params
