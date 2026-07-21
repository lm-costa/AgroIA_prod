import json
import os
import shutil
import time

import nlopt
import numpy as np
import pandas as pd
from pcse.base import ParameterProvider
from pcse.input import CABOFileReader, YAMLAgroManagementReader
from pcse.input import YAMLCropDataProvider
from pcse.input.sitedataproviders import WOFOST72SiteDataProvider
from pcse.models import Wofost72_WLP_FD

from SensitivityAnalyzer import NetCDFDataLoader
from utils import setup_paths, update_agro_management_file, map_info, WOFOST_bounds


class WOFOSTOptimizer:
    """
    Classe para otimização de parâmetros do WOFOST usando NLOPT,
    separada por cluster usando dados NetCDF.
    """

    # Parâmetros por cluster (do Morris Screen)
    CLUSTER_PARAMS = {
        0.0: ['TMPFTB020', 'AMAXTB082', 'TSUM1', 'TDWI', 'DVSI', 'SPAN', 'DVSEND', 'SLATB100', 'RRI', 'RDMCR', 'TMPFTB035', 'CVO', 'AMAXTB200', 'SLATB064', 'TBASE', 'SLATB029', 'RFSETB200', 'RMS', 'IDSL', 'CFET', 'RMO', 'AMAXTB014', 'DLC', 'TSUM2', 'RFSETB000', 'DEPNR', 'RDI', 'SLATB000', 'RML', 'PERDL', 'TMPFTB008', 'CVS', 'SLATB021', 'CVL', 'RMR', 'RGRLAI', 'Q10', 'SLATB200', 'AMAXTB000', 'CVR', 'DLO', 'IAIRDU', 'IOX', 'TMPFTB000'],
        1.0: ['TMPFTB020', 'AMAXTB082', 'TSUM1', 'TDWI', 'SPAN', 'DVSI', 'SLATB100', 'TMPFTB035', 'SLATB064', 'AMAXTB200', 'DVSEND', 'TBASE', 'IDSL', 'RRI', 'CVO', 'SLATB029', 'RFSETB200', 'RMS', 'AMAXTB014', 'DLC', 'SLATB000', 'RMO', 'CFET', 'RFSETB000', 'RDMCR', 'RDI', 'TSUM2', 'TMPFTB008', 'DEPNR', 'RML', 'SLATB021', 'CVS', 'PERDL', 'RMR', 'CVL', 'RGRLAI', 'DLO', 'Q10', 'SLATB200', 'CVR', 'AMAXTB000', 'IAIRDU', 'IOX', 'TMPFTB045'],
        2.0: ['AMAXTB082', 'TMPFTB020', 'TDWI', 'TMPFTB035', 'SLATB100', 'SPAN', 'TSUM1', 'DVSEND', 'AMAXTB200', 'DVSI', 'CVO', 'SLATB064', 'DLC', 'RFSETB200', 'RMO', 'IDSL', 'RMS', 'SLATB029', 'AMAXTB014', 'TBASE', 'RRI', 'RFSETB000', 'TSUM2', 'SLATB000', 'RML', 'RDMCR', 'CFET', 'CVS', 'SLATB021', 'RDI', 'RMR', 'RGRLAI', 'CVL', 'DLO', 'DEPNR', 'PERDL', 'IOX', 'IAIRDU', 'SLATB200', 'AMAXTB000', 'Q10', 'CVR', 'TMPFTB008', 'TMPFTB045'],
        3.0: ['TMPFTB020', 'AMAXTB082', 'TSUM1', 'TDWI', 'DVSI', 'SPAN', 'SLATB100', 'SLATB064', 'AMAXTB200', 'DVSEND', 'CVO', 'TBASE', 'SLATB029', 'AMAXTB014', 'TSUM2', 'DLC', 'TMPFTB035', 'RMS', 'RFSETB200', 'IDSL', 'RMO', 'RFSETB000', 'TMPFTB008', 'SLATB000', 'RML', 'RMR', 'DLO', 'CVS', 'Q10', 'SLATB021', 'RRI', 'CVL', 'RGRLAI', 'SLATB200', 'CFET', 'AMAXTB000', 'RDI', 'DEPNR', 'CVR', 'RDMCR', 'PERDL', 'TMPFTB000', 'IAIRDU', 'IOX'],
        4.0: ['TMPFTB035', 'AMAXTB082', 'RRI', 'TSUM1', 'TDWI', 'DVSEND', 'RDI', 'SPAN', 'DVSI', 'AMAXTB200', 'RDMCR', 'SLATB100', 'DLC', 'TMPFTB045', 'SLATB064', 'IDSL', 'CFET', 'RFSETB200', 'SLATB029', 'AMAXTB014', 'SLATB000', 'RMO', 'RMS', 'CVO', 'TMPFTB020', 'DEPNR', 'DLO', 'RFSETB000', 'RML', 'TSUM2', 'SLATB021', 'IAIRDU', 'RMR', 'Q10', 'PERDL', 'TBASE', 'AMAXTB000', 'CVL', 'CVS', 'IOX', 'RGRLAI', 'CVR', 'SLATB200']
    }

    def __init__(self, paths, nc_loader, algorithm=nlopt.LN_BOBYQA, max_eval=5000):
        """
        Inicializa o otimizador.

        Args:
            paths: Dicionário com caminhos dos arquivos
            nc_loader: Instância de NetCDFDataLoader
            algorithm: Algoritmo NLOPT a ser usado
            max_eval: Número máximo de avaliações
        """
        self.paths = paths
        self.nc_loader = nc_loader
        self.algorithm = algorithm
        self.max_eval = max_eval
        self.n_eval = 0
        self.best_params = None
        self.best_rmse = float('inf')

        import logging
        self.logger = logging.getLogger('pcse')
        self.logger.setLevel(logging.ERROR)

    def create_weather_data_provider(self, weather_df, latitude, longitude, elevation=None):
        """Reutiliza o metodo do MorrisScreeningAnalyzer"""
        from SensitivityAnalyzer import MorrisScreeningAnalyzer

        # Criar instância temporária apenas para usar o metodo
        temp_analyzer = MorrisScreeningAnalyzer({}, self.paths, self.nc_loader)
        return temp_analyzer.create_weather_data_provider(weather_df, latitude, longitude, elevation)

    def setup_optimization_problem(self, cluster_id):
        """
        Configura o problema de otimização para um cluster específico.

        Args:
            cluster_id: ID do cluster

        Returns:
            Dicionário com configuração do problema
        """
        params = self.CLUSTER_PARAMS.get(cluster_id, [])
        bounds = WOFOST_bounds(params)

        return {
            'params': params,
            'bounds': bounds,
            'lower': [bounds[p][0] for p in params],
            'upper': [bounds[p][1] for p in params],
            'n_params': len(params)
        }

    @staticmethod
    def extract_model_params(X, param_names):
        """
        Extrai parâmetros do modelo a partir do vetor X.

        Args:
            X: Vetor com valores dos parâmetros
            param_names: Lista com nomes dos parâmetros

        Returns:
            Dicionário com parâmetros para o modelo
        """
        return {param_names[i]: X[i] for i in range(len(param_names))}

    def run_wofost_simulation(self, model_params, parameters, weather, agromanagement):
        """
        Executa simulação WOFOST.

        Args:
            model_params: Dicionário com parâmetros do modelo
            parameters: ParameterProvider
            weather: WeatherDataProvider
            agromanagement: AgroManagement

        Returns:
            Array com produtividades simuladas por ano
        """
        try:
            # Configurar tabelas de parâmetros (igual ao MorrisScreeningAnalyzer)
            self._configure_parameter_tables(model_params, parameters)

            # Inicializar e executar modelo
            wofost = Wofost72_WLP_FD(parameters, weather, agromanagement)
            wofost.run_till_terminate()

            output = wofost.get_output()
            if output:
                final_output = output[-1]
                yield_kg_ha = final_output.get('TWSO', 0) * 1000
                return yield_kg_ha
            else:
                return np.nan

        except Exception as e:
            self.logger.error(f"Erro na simulação: {e}")
            return np.nan

    def _configure_parameter_tables(self, model_params, parameters):
        """Configura tabelas de parâmetros (SLATB, AMAXTB, etc.)"""

        # SLATB
        if any(key.startswith('SLATB') for key in model_params):
            slatb_values = []
            for dvs, key in [(0.00, 'SLATB000'), (0.21, 'SLATB021'), (0.29, 'SLATB029'),
                             (0.64, 'SLATB064'), (1.00, 'SLATB100'), (2.00, 'SLATB200')]:
                if key in model_params:
                    slatb_values.extend([dvs, model_params[key]])
            if slatb_values:
                parameters.set_override("SLATB", slatb_values)

        # AMAXTB
        if any(key.startswith('AMAXTB') for key in model_params):
            amaxtb_values = []
            for dvs, key in [(0.00, 'AMAXTB000'), (0.14, 'AMAXTB014'),
                             (0.82, 'AMAXTB082'), (2.00, 'AMAXTB200')]:
                if key in model_params:
                    amaxtb_values.extend([dvs, model_params[key]])
            if amaxtb_values:
                parameters.set_override('AMAXTB', amaxtb_values)

        # TMPFTB
        if any(key.startswith('TMPFTB') for key in model_params):
            tmpftb_values = []
            for tmp, key in [(0.00, 'TMPFTB000'), (8.00, 'TMPFTB008'), (20.0, 'TMPFTB020'),
                             (35.0, 'TMPFTB035'), (45.0, 'TMPFTB045')]:
                if key in model_params:
                    tmpftb_values.extend([tmp, model_params[key]])
            if tmpftb_values:
                parameters.set_override("TMPFTB", tmpftb_values)

        # RFSETB
        if any(key.startswith('RFSETB') for key in model_params):
            rfsetb_values = []
            for dvs, key in [(0, 'RFSETB000'), (2, 'RFSETB200')]:
                if key in model_params:
                    rfsetb_values.extend([dvs, model_params[key]])
            if rfsetb_values:
                parameters.set_override("RFSETB", rfsetb_values)

        # Parâmetros simples
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

        # Validação DLO > DLC
        if 'DLC' in model_params and 'DLO' in model_params:
            if model_params['DLO'] <= model_params['DLC']:
                dlc_adjusted = model_params['DLC']
                dlo_adjusted = dlc_adjusted + 0.1
                parameters.set_override('DLC', dlc_adjusted)
                parameters.set_override('DLO', dlo_adjusted)

    def objective_function(self, X, grad, context):
                    """
                    Função objetivo: minimizar erro entre simulado normalizado e observado normalizado.
                    """
                    model_params = self.extract_model_params(X, context['param_names'])

                    # 1) Simular produtividade (já normalizada virtualmente pela calibração)
                    yield_sim = self.run_wofost_simulation(
                        model_params,
                        context['parameters'],
                        context['weather'],
                        context['agromanagement']
                    )

                    if np.isnan(yield_sim):
                        return 1e10

                    # 2) Comparar diretamente com dyield (que JÁ está normalizado)
                    dyield_target = context['dyield_target']  # Valor normalizado do .nc

                    error = abs(yield_sim - dyield_target)
                    return error


    def optimize_point_hierarchical(self, nc_file, cluster_id, max_params_per_stage=5):
            """
            Otimiza parâmetros de forma hierárquica (por ordem de importância).
            Interrompe quando erro absoluto = 0.0 entre etapas.

            Args:
                nc_file: Caminho do arquivo NetCDF
                cluster_id: ID do cluster
                max_params_per_stage: Número máximo de parâmetros a adicionar por etapa

            Returns:
                DataFrame com resultados da otimização por ano
            """
            point_info, weather_df = self.nc_loader.load_point_data(nc_file)
            point_id = point_info['point_id']

            print(f"\n🎯 Otimizando Point {point_id} (Cluster {cluster_id}) - MODO HIERÁRQUICO")

            # Preparar dados
            LAT = point_info['latitude']
            LON = point_info['longitude']
            location = point_info.get('country', 'Brazil')
            elevation = point_info.get('elevation', np.nan)

            # Carregar parâmetros base
            cropfile = YAMLCropDataProvider(fpath=self.paths['CROP'])
            cropfile.set_active_crop('maize', "Maize_VanHeemst_1988")

            countries_info = map_info()
            soil_file = countries_info.get(location, {}).get('soil_file', 'ec4.new')
            soil_path = os.path.join(os.path.dirname(self.paths['SOIL']), soil_file)
            soildata = CABOFileReader(fname=soil_path)
            sitedata = WOFOST72SiteDataProvider(WAV=100)

            # Extrair anos disponíveis
            weather_df['year'] = pd.to_datetime(weather_df['date']).dt.year
            years_with_data = weather_df[weather_df['dyield'].notna()]['year'].unique()

            print(f"📅 Anos com dados: {len(years_with_data)}")

            results_list = []

            # Obter lista ordenada de parâmetros (por importância)
            all_params = self.CLUSTER_PARAMS[cluster_id]

            # Otimizar para cada ano
            for year in sorted(years_with_data):
                print(f"\n{'='*60}")
                print(f"📊 SAFRA {year}")
                print(f"{'='*60}")

                # Filtrar dados do ano
                year_data = weather_df[weather_df['year'] == year].copy()
                dyield_obs = year_data['dyield'].dropna()

                if len(dyield_obs) == 0:
                    print(f"⚠️  Sem dyield válido para {year}")
                    continue

                dyield_target = dyield_obs.values
                start_date = pd.to_datetime(year_data['date'].iloc[0])

                # Criar weather provider
                weather = self.create_weather_data_provider(year_data, LAT, LON, elevation)

                # Atualizar agromanagement
                agro_path_temp = self.paths['AGRO'] + f'_opt_{point_id}_{year}.yaml'
                shutil.copy(self.paths['AGRO'], agro_path_temp)
                update_agro_management_file(agro_path_temp, start_date, location)

                agromanagement = YAMLAgroManagementReader(agro_path_temp)
                parameters = ParameterProvider(cropdata=cropfile, soildata=soildata, sitedata=sitedata)

                # Dicionário para armazenar valores otimizados acumulados
                optimized_params = {}
                min_error = float('inf')
                optimization_stopped_early = False

                # Otimização incremental
                n_stages = (len(all_params) + max_params_per_stage - 1) // max_params_per_stage

                for stage in range(1, n_stages + 1):
                    # Parâmetros a otimizar nesta etapa
                    end_idx = min(stage * max_params_per_stage, len(all_params))
                    params_to_optimize = all_params[:end_idx]

                    print(f"\n🔧 Etapa {stage}/{n_stages}: Otimizando {len(params_to_optimize)} parâmetros")
                    print(f"   Novos: {params_to_optimize[-max_params_per_stage:]}")

                    # Configurar problema de otimização
                    problem = self.setup_optimization_problem(cluster_id)

                    # Filtrar apenas os parâmetros da etapa atual
                    stage_bounds = {p: problem['bounds'][p] for p in params_to_optimize}
                    stage_lower = [stage_bounds[p][0] for p in params_to_optimize]
                    stage_upper = [stage_bounds[p][1] for p in params_to_optimize]

                    # Criar otimizador
                    opt = nlopt.opt(self.algorithm, len(params_to_optimize))
                    opt.set_lower_bounds(stage_lower)
                    opt.set_upper_bounds(stage_upper)

                    # Contexto com parâmetros fixos
                    context = {
                        'parameters': parameters,
                        'weather': weather,
                        'agromanagement': agromanagement,
                        'param_names': params_to_optimize,
                        'fixed_params': optimized_params.copy(),
                        'dyield_target': dyield_target
                    }

                    opt.set_min_objective(lambda X, grad: self.objective_function_hierarchical(X, grad, context))

                    # Critérios de parada
                    max_eval_stage = self.max_eval // n_stages
                    opt.set_maxeval(max_eval_stage)
                    opt.set_xtol_rel(1e-5 if stage <= 3 else 1e-4)
                    opt.set_ftol_abs(0.001)  # Parar se erro absoluto < 1 gramma/ha

                    # Ponto inicial
                    x0 = []
                    for param in params_to_optimize:
                        if param in optimized_params:
                            x0.append(optimized_params[param])
                        else:
                            lb, ub = stage_bounds[param]
                            x0.append((lb + ub) / 2)

                    try:
                        # Executar otimização da etapa
                        x_opt = opt.optimize(x0)
                        min_error = opt.last_optimum_value()

                        # Atualizar parâmetros otimizados
                        for i, param_name in enumerate(params_to_optimize):
                            optimized_params[param_name] = x_opt[i]

                        print(f"   ✅ Erro: {min_error:.4f} kg/ha")

                        if min_error < 0.050:
                            print('ERRO < 0.05 kg/ha, interrompendo otimização.')
                            optimization_stopped_early = True
                            break

                    except Exception as e:
                        print(f"   ❌ Erro na etapa {stage}: {e}")
                        break

                # Salvar resultado do ano
                result = {
                    'point_id': point_id,
                    'cluster_id': cluster_id,
                    'year': year,
                    'latitude': LAT,
                    'longitude': LON,
                    'dyield_obs': np.mean(dyield_target),
                    'final_error': min_error,
                    'n_stages': stage if optimization_stopped_early else n_stages,
                    'stopped_early': optimization_stopped_early
                }

                # Adicionar todos os parâmetros otimizados
                result.update(optimized_params)

                results_list.append(result)

                status = "PERFEITO ✨" if optimization_stopped_early else f"Completado"
                print(f"\n🎉 Ano {year}: {status} - Erro final = {min_error:.4f} kg/ha ({result['n_stages']} etapas)")

                # Limpar arquivo temporário
                if os.path.exists(agro_path_temp):
                    os.remove(agro_path_temp)

            if not results_list:
                return None

            # Converter para DataFrame
            results_df = pd.DataFrame(results_list)

            # Salvar resultados
            output_file = os.path.join(
                self.paths['OPTIMIZATION'],
                f"OPT_HIER_cluster{cluster_id}_point{point_id}.csv"
            )
            results_df.to_csv(output_file, index=False)
            print(f"\n💾 Resultados hierárquicos salvos em {output_file}")

            return results_df

    def objective_function_hierarchical(self, X, grad, context):
            """
            Função objetivo para otimização hierárquica.
            Combina parâmetros fixos (já otimizados) com os da etapa atual.
            """
            # Combinar parâmetros fixos com os da etapa atual
            model_params = context['fixed_params'].copy()

            for i, param_name in enumerate(context['param_names']):
                model_params[param_name] = X[i]

            # Simular produtividade
            yield_sim = self.run_wofost_simulation(
                model_params,
                context['parameters'],
                context['weather'],
                context['agromanagement']
            )

            if np.isnan(yield_sim):
                return 1e10

            # Comparar diretamente com dyield (que JÁ está normalizado)
            dyield_target_mean = np.mean(context['dyield_target'])

            error = abs(yield_sim - dyield_target_mean)
            return error

    def run_optimization(self):
        """
        Executa a otimização para todos os pontos, separados por cluster.
        """
        print("=== INICIANDO OTIMIZAÇÃO DE PARÂMETROS ===")

        # Obter todos os arquivos NetCDF
        all_files = self.nc_loader.nc_files

        # Agrupar por cluster
        files_by_cluster = {}
        for nc_file in all_files:
            point_info = self.nc_loader.get_point_info(nc_file)
            cluster_id = point_info['cluster_id']

            if cluster_id not in files_by_cluster:
                files_by_cluster[cluster_id] = []
            files_by_cluster[cluster_id].append(nc_file)

        # Executar otimização por cluster
        all_results = []

        for cluster_id in sorted(files_by_cluster.keys()):
            files = files_by_cluster[cluster_id]
            print(f"\n{'='*60}")
            print(f"CLUSTER {cluster_id}: {len(files)} pontos")
            print(f"{'='*60}")

            for i, nc_file in enumerate(files, 1):
                print(f"\n[{i}/{len(files)}]")
                result = self.optimize_point_hierarchical(nc_file, cluster_id, max_params_per_stage=5)
                if result is not None:
                    all_results.append(result)

        # Salvar resumo
        summary_file = os.path.join(self.paths['OPTIMIZATION'], 'optimization_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✅ Otimização concluída! Resultados salvos em {self.paths['OPTIMIZATION']}")
        return all_results


def main():
    """Função principal"""
    start_time = time.time()

    # Configurar caminhos
    paths = setup_paths()

    # Carregar dados NetCDF
    nc_loader = NetCDFDataLoader(paths['COMPLETO'])

    # Criar otimizador
    optimizer = WOFOSTOptimizer(
        paths=paths,
        nc_loader=nc_loader,
        algorithm=nlopt.LN_COBYLA,
        max_eval=5000
    )

    # Executar otimização
    results = optimizer.run_optimization()

    elapsed = time.time() - start_time
    print(f"\n⏱️  Tempo total: {elapsed/3600:.2f} horas")


if __name__ == '__main__':
    main()