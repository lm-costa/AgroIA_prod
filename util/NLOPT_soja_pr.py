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
import traceback

import numpy as np
import pandas as pd

from pcse.base import ParameterProvider
from pcse.input import CABOFileReader, YAMLAgroManagementReader
from pcse.input import YAMLCropDataProvider
from pcse.input.sitedataproviders import WOFOST72SiteDataProvider
from pcse.models import Wofost72_WLP_FD

from NLOPT_MultiYear import WOFOSTMultiYearOptimizer
from utils_soja_pr import (
    clamp_rdi_rdmcr,
    map_info_soja_pr,
    safe_elevation,
    safra_ano_colheita,
    to_dry_matter,
    update_agro_management_file_soja_pr,
)


class SoyWOFOSTMultiYearOptimizerPR(WOFOSTMultiYearOptimizer):
    """Calibracao multi-anual do WOFOST para soja/PR (BR-DWGD + IBGE)."""

    CROP_NAME = 'soybean'
    VARIETY_NAME = 'Soybean_VanHeemst_1988'

    # Serie de produtividade observada usada como alvo da calibracao.
    # 'yield' e a produtividade observada direta (IBGE/SIDRA, kg/ha), que
    # preserva a variabilidade temporal real (clima, manejo, tecnologia).
    # 'dyield' e a versao sem tendencia (detrending), pensada para estudo
    # cientifico da resposta climatica e nao para calibrar um modelo que deve
    # reproduzir a variabilidade entre safras. A chave de saida continua
    # 'dyield_target' por compatibilidade com o restante do pipeline.
    OBSERVED_YIELD_COLUMN = 'yield'

    # Quantas falhas de simulacao detalhar (params + traceback completo) antes
    # de voltar a so logar a mensagem de erro resumida. Diagnostico -- ver
    # docstring de run_wofost_simulation abaixo.
    MAX_DEBUG_FAILURES = 3

    def __init__(self, paths, nc_loader, cluster_params, algorithm=None, max_eval=3000):
        import nlopt
        if algorithm is None:
            algorithm = nlopt.LN_BOBYQA
        super().__init__(paths, nc_loader, algorithm, max_eval)
        # Sobrescreve o CLUSTER_PARAMS herdado (calibrado para milho) pelo
        # ranking especifico de soja/PR obtido na etapa de Sensitivity Analysis.
        self.CLUSTER_PARAMS = cluster_params
        self._debug_failures_logged = 0

    @staticmethod
    def extract_model_params(X, param_names):
        model_params = WOFOSTMultiYearOptimizer.extract_model_params(X, param_names)
        return clamp_rdi_rdmcr(model_params)

    # Joelhos (x, nome do parametro) de cada tabela calibravel -- mesma
    # parametrizacao de NLOPT._configure_parameter_tables (milho).
    TABLE_KNOTS = {
        'SLATB': [(0.00, 'SLATB000'), (0.21, 'SLATB021'), (0.29, 'SLATB029'),
                  (0.64, 'SLATB064'), (1.00, 'SLATB100'), (2.00, 'SLATB200')],
        'AMAXTB': [(0.00, 'AMAXTB000'), (0.14, 'AMAXTB014'), (0.82, 'AMAXTB082'), (2.00, 'AMAXTB200')],
        'TMPFTB': [(0.0, 'TMPFTB000'), (8.0, 'TMPFTB008'), (20.0, 'TMPFTB020'),
                   (35.0, 'TMPFTB035'), (45.0, 'TMPFTB045')],
        'RFSETB': [(0, 'RFSETB000'), (2, 'RFSETB200')],
    }

    @staticmethod
    def _default_table(parameters, table):
        """Tabela padrao da variedade como (xs, ys), ou None se indisponivel."""
        try:
            values = np.asarray(parameters._cropdata[table], dtype=float)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        return values[0::2], values[1::2]

    def default_param_value(self, name, parameters):
        """
        Valor padrao da variedade para o parametro `name` (escalar ou joelho de
        tabela, interpolado na tabela padrao), ou None se nao houver.
        """
        for table, knots in self.TABLE_KNOTS.items():
            for x, key in knots:
                if key == name:
                    default = self._default_table(parameters, table)
                    return None if default is None else float(np.interp(x, *default))
        try:
            value = parameters._cropdata[name]
            value = float(value[0] if isinstance(value, (list, tuple)) else value)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            return None
        return value if np.isfinite(value) else None

    def _configure_parameter_tables(self, model_params, parameters):
        """
        Como a versao do milho (NLOPT.py), mas sem descartar os pontos de tabela
        nao calibrados: o base so envia ao override os joelhos presentes em
        `model_params`, o que SUBSTITUI a tabela inteira da variedade por poucos
        pontos (medido: AMAXTB so com {0.82, 2.0} reduz o TWSO padrao em ~58%).
        Aqui cada tabela calibrada = tabela padrao + os joelhos calibrados
        (inseridos; os pontos originais, ex. AMAXTB em DVS 1.7, sao mantidos),
        entao uma tabela sem nenhum joelho calibrado, ou com joelhos iguais ao
        padrao, reproduz exatamente a variedade.
        """
        knot_keys = {key for knots in self.TABLE_KNOTS.values() for _, key in knots}
        for table, knots in self.TABLE_KNOTS.items():
            calibrated = [(x, model_params[key]) for x, key in knots if key in model_params]
            if not calibrated:
                continue
            default = self._default_table(parameters, table)
            if default is None:
                continue
            xs0, ys0 = default
            xs = sorted({round(float(x), 6) for x in xs0} | {round(float(x), 6) for x, _ in calibrated})
            ys = [float(np.interp(x, xs0, ys0)) for x in xs]
            for x, value in calibrated:
                ys[xs.index(round(float(x), 6))] = float(value)
            parameters.set_override(table, [v for pair in zip(xs, ys) for v in pair])
        super()._configure_parameter_tables(
            {k: v for k, v in model_params.items() if k not in knot_keys}, parameters)

    def run_wofost_simulation(self, model_params, parameters, weather, agromanagement):
        """
        Igual a WOFOSTOptimizer.run_wofost_simulation (util/NLOPT.py), exceto
        que loga os parametros exatos e o traceback completo nas primeiras
        MAX_DEBUG_FAILURES falhas -- diagnostico usado para achar a causa real
        do 'NoneType' object has no attribute 'add_variable' (era a janela de
        clima cortada por ano civil em prepare_multiyear_context, ja
        corrigido abaixo; mantido como rede de seguranca para falhas futuras).
        """
        try:
            self._configure_parameter_tables(model_params, parameters)
            wofost = Wofost72_WLP_FD(parameters, weather, agromanagement)
            wofost.run_till_terminate()

            output = wofost.get_output()
            if output:
                final_output = output[-1]
                # TWSO do PCSE ja esta em kg/ha (mesma unidade do IBGE/SIDRA
                # 5457 var. 112); multiplicar por 1000 inflava a simulacao.
                return final_output.get('TWSO', 0)
            return np.nan

        except Exception as e:
            if self._debug_failures_logged < self.MAX_DEBUG_FAILURES:
                self._debug_failures_logged += 1
                print(f"\n[DEBUG {self._debug_failures_logged}/{self.MAX_DEBUG_FAILURES}] Falha na simulacao: {e}")
                print(f"[DEBUG] model_params: {model_params}")
                traceback.print_exc()
                print()
            self.logger.error(f"Erro na simulação: {e}")
            return np.nan

    def prepare_multiyear_context(self, point_info, weather_df, cluster_id):
        """
        Monta o clima/agromanagement de cada safra com produtividade valida.

        O alvo de calibracao usa a produtividade COM tendencia (coluna
        'yield' do NetCDF, publicada pelo IBGE) e nao a serie 'dyield'
        (detrended, com a tendencia tecnologica removida e ancorada no ano
        mais recente). A chave do dicionario continua 'dyield_target' por
        compatibilidade com o resto do pipeline (NLOPT_regional_soja_pr.py
        usa esse mesmo nome), mas o valor vem de 'yield'.

        IMPORTANTE: a janela de clima de cada safra e selecionada por
        INTERVALO DE DATAS real do ciclo (semeadura ate semeadura+duracao),
        nao por ano civil. A versao anterior filtrava `weather_df['year'] ==
        ano` (ano civil da data), o que corta o provider de clima em 31/dez
        -- com semeadura em nov (2o semestre), o ciclo atravessa a virada do
        ano, e o WOFOST quebra ('NoneType' object has no attribute
        'add_variable') assim que a simulacao passa de 31/dez sem dado
        climatico disponivel. Isso ja acontecia com out/200d tambem (so que
        com folga ate dezembro), o que explica os ajustes suspeitosamente
        rapidos/bons vistos antes: eram combinacoes de parametros com
        fenologia curta o bastante pra "escapar" do corte de ano civil, nao
        um ajuste agronomico real.
        """
        LAT = point_info['latitude']
        LON = point_info['longitude']
        elevation = safe_elevation(point_info.get('elevation', np.nan))

        calendar_info = map_info_soja_pr()
        soil_file = calendar_info['soil_file']
        sowing_month = calendar_info['sowing_month']
        duration_days = calendar_info['max_duration']

        cropfile = YAMLCropDataProvider(fpath=self.paths['CROP'])
        cropfile.set_active_crop(self.CROP_NAME, self.VARIETY_NAME)

        soil_path = os.path.join(os.path.dirname(self.paths['SOIL']), soil_file)
        soildata = CABOFileReader(fname=soil_path)
        sitedata = WOFOST72SiteDataProvider(WAV=100)

        weather_df = weather_df.copy()
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        weather_df = weather_df.sort_values('date').reset_index(drop=True)
        weather_df['ano_safra'] = weather_df['date'].apply(safra_ano_colheita)

        col = self.OBSERVED_YIELD_COLUMN
        safras_com_dyield = sorted(weather_df.loc[weather_df[col].notna(), 'ano_safra'].unique())

        years_data = []

        for safra in safras_com_dyield:
            dyield_obs = weather_df.loc[weather_df['ano_safra'] == safra, col].dropna()

            if len(dyield_obs) == 0:
                continue

            # Ano civil de semeadura: se o mes de semeadura cai no 2o
            # semestre (jul-dez), a safra e colhida no ano seguinte (mesma
            # convencao de safra_ano_colheita); senao, semeadura e colheita
            # caem no mesmo ano civil.
            ano_semeadura = safra - 1 if sowing_month >= 7 else safra
            crop_start = pd.Timestamp(year=int(ano_semeadura), month=sowing_month, day=calendar_info['sowing_day'])
            crop_end = crop_start + pd.Timedelta(days=duration_days)

            janela = weather_df[(weather_df['date'] >= crop_start) & (weather_df['date'] <= crop_end)]

            if janela.empty or janela['date'].max() < crop_end:
                continue

            weather = self.create_weather_data_provider(janela, LAT, LON, elevation)

            agro_path_temp = f"{self.paths['AGRO']}_temp_{cluster_id}_{safra}.yaml"
            shutil.copy(self.paths['AGRO'], agro_path_temp)
            update_agro_management_file_soja_pr(agro_path_temp, crop_start)
            agromanagement = YAMLAgroManagementReader(agro_path_temp)

            parameters = ParameterProvider(cropdata=cropfile, soildata=soildata, sitedata=sitedata)

            years_data.append({
                'year': safra,
                'weather': weather,
                'agromanagement': agromanagement,
                'parameters': parameters,
                'dyield_target': to_dry_matter(np.mean(dyield_obs.values)),
                'agro_temp_file': agro_path_temp
            })

        return years_data


# IDSL, IAIRDU e IOX sao flags binarias/categoricas internas do WOFOST (ex:
# IDSL escolhe o submodelo de fenologia -- so temperatura, +fotoperiodo,
# +vernalizacao), nao parametros continuos de verdade. WOFOST_bounds() os
# define como intervalo continuo (0, 1) por serem reaproveitados do pipeline
# de milho, entao o NLOPT testa valores fracionarios (ex: 0.63) que nao
# correspondem a nenhum estado valido do modelo e derrubam a simulacao
# ('NoneType' object has no attribute 'add_variable'). Excluidos aqui do
# conjunto otimizavel: ficam com o valor padrao definido na propria
# variedade de cultura (Soybean_VanHeemst_1988), em vez de serem calibrados.
EXCLUDED_PARAMS = {'IDSL', 'IAIRDU', 'IOX'}


def load_cluster_params_from_ranking(ranking_json_path, top_n=44):
    """
    Le o ranking de sensibilidade por cluster salvo por 6.SA_Soja_PR.ipynb e
    monta o dicionario {cluster_id: [param_names ordenados por mu_star]} no
    mesmo formato que WOFOSTOptimizer.CLUSTER_PARAMS usa para o milho.

    Espera um JSON no formato:
        {"<cluster_id>": [{"Rank":1, "parameter": "TSUM1", "mu_star": ..., "sigma": ...}, ...], ...}

    Parametros em EXCLUDED_PARAMS sao removidos do ranking antes de aplicar
    top_n, entao nunca entram no conjunto otimizado pelo NLOPT.
    """
    import json

    with open(ranking_json_path, 'r') as f:
        ranking = json.load(f)

    cluster_params = {}
    for cluster_id_str, params_ranked in ranking.items():
        cluster_id = float(cluster_id_str)
        names = [
            p['parameter'] for p in params_ranked
            if p['parameter'] not in EXCLUDED_PARAMS
        ][:top_n]
        cluster_params[cluster_id] = names

    return cluster_params
