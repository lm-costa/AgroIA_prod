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

from SensitivityAnalyzer import NetCDFDataLoader
from NLOPT import WOFOSTOptimizer
from utils import setup_paths, update_agro_management_file, map_info


# Constantes de configuração
EARLY_STOPPING_RMSE_THRESHOLD = 100  # kg/ha - Interromper otimização se RMSE < 100 kg/ha
XTOL_REL_EARLY_STAGES = 1e-5  # Tolerância relativa para primeiras 3 etapas
XTOL_REL_LATER_STAGES = 1e-4  # Tolerância relativa para etapas posteriores
FTOL_ABS = 0.1  # kg/ha - Tolerância absoluta (equivalente a 100 g/ha)
EARLY_STAGE_THRESHOLD = 3  # Número de etapas consideradas "iniciais"


class WOFOSTMultiYearOptimizer(WOFOSTOptimizer):
    """
    Otimizador WOFOST para calibração multi-anual (40 anos simultâneos).
    Herda de WOFOSTOptimizer para reutilizar métodos auxiliares.
    
    Esta classe implementa calibração usando TODOS os anos simultaneamente como
    múltiplos targets, reduzindo drasticamente o tempo computacional comparado
    à calibração anual.
    
    Ganho esperado: ~40x redução no tempo total de processamento.
    """

    def __init__(self, paths, nc_loader, algorithm=nlopt.LN_BOBYQA, max_eval=5000):
        """
        Inicializa o otimizador multi-anual.

        Args:
            paths: Dicionário com caminhos dos arquivos
            nc_loader: Instância de NetCDFDataLoader
            algorithm: Algoritmo NLOPT a ser usado (default: LN_BOBYQA)
            max_eval: Número máximo de avaliações por otimização
        """
        super().__init__(paths, nc_loader, algorithm, max_eval)
        
    def prepare_multiyear_context(self, point_info, weather_df, cluster_id):
        """
        Prepara dados de todos os anos para otimização multi-anual.
        
        Cria uma lista com dados de cada ano incluindo:
        - WeatherDataProvider específico do ano
        - AgroManagement configurado para o ano
        - ParameterProvider (cópia independente)
        - dyield observado (target)
        
        Args:
            point_info: Metadados do ponto (dict com latitude, longitude, etc.)
            weather_df: DataFrame com dados climáticos + dyield
            cluster_id: ID do cluster
            
        Returns:
            list: Lista de dicionários, um para cada ano com dados válidos
                [
                    {
                        'year': int,
                        'weather': WeatherDataProvider,
                        'agromanagement': YAMLAgroManagementReader,
                        'parameters': ParameterProvider,
                        'dyield_target': float,
                        'agro_temp_file': str
                    },
                    ...
                ]
        """
        LAT = point_info['latitude']
        LON = point_info['longitude']
        location = point_info.get('country', 'Brazil')
        elevation = point_info.get('elevation', np.nan)
        
        # Carregar parâmetros base (solo, sítio, cultura)
        cropfile = YAMLCropDataProvider(fpath=self.paths['CROP'])
        cropfile.set_active_crop('maize', "Maize_VanHeemst_1988")
        
        countries_info = map_info()
        soil_file = countries_info.get(location, {}).get('soil_file', 'ec4.new')
        soil_path = os.path.join(os.path.dirname(self.paths['SOIL']), soil_file)
        soildata = CABOFileReader(fname=soil_path)
        sitedata = WOFOST72SiteDataProvider(WAV=100)
        
        # Processar cada ano
        weather_df['year'] = pd.to_datetime(weather_df['date']).dt.year
        years_with_data = weather_df[weather_df['dyield'].notna()]['year'].unique()
        
        years_data = []
        
        for year in sorted(years_with_data):
            year_data_df = weather_df[weather_df['year'] == year].copy()
            dyield_obs = year_data_df['dyield'].dropna()
            
            if len(dyield_obs) == 0:
                continue
            
            # Criar providers específicos do ano
            weather = self.create_weather_data_provider(year_data_df, LAT, LON, elevation)
            
            # Agromanagement temporário
            agro_path_temp = f"{self.paths['AGRO']}_temp_{cluster_id}_{year}.yaml"
            shutil.copy(self.paths['AGRO'], agro_path_temp)
            start_date = pd.to_datetime(year_data_df['date'].iloc[0])
            update_agro_management_file(agro_path_temp, start_date, location)
            agromanagement = YAMLAgroManagementReader(agro_path_temp)
            
            # ParameterProvider (cópia independente para cada ano)
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

    def objective_function_multiyear(self, X, grad, context):
        """
        Função objetivo que minimiza erro agregado de TODOS os anos.
        
        Estratégia:
        1. Para cada ano (1-40):
           - Executar simulação WOFOST com parâmetros X
           - Calcular erro vs. dyield observado
        2. Agregar erros usando RMSE médio:
           RMSE_total = sqrt(mean([erro_ano1², erro_ano2², ..., erro_ano40²]))
        
        Args:
            X: Vetor de parâmetros a otimizar
            grad: Gradiente (não usado em LN_COBYLA/BOBYQA)
            context: Dicionário contendo:
                - years_data: Lista com dados de cada ano
                - param_names: Lista de parâmetros sendo otimizados
                - fixed_params: Parâmetros fixos (se otimização hierárquica)
        
        Returns:
            float: RMSE agregado (kg/ha)
        """
        # Extrair parâmetros do modelo
        model_params = self.extract_model_params(X, context['param_names'])
        
        # Combinar com parâmetros fixos (se existirem - otimização hierárquica)
        if 'fixed_params' in context:
            model_params.update(context['fixed_params'])
        
        squared_errors = []
        
        # Simular para cada ano
        for year_data in context['years_data']:
            yield_sim = self.run_wofost_simulation(
                model_params,
                year_data['parameters'],
                year_data['weather'],
                year_data['agromanagement']
            )
            
            if np.isnan(yield_sim):
                return 1e10  # Penalidade para simulações inválidas
            
            # Erro quadrático
            error = (yield_sim - year_data['dyield_target']) ** 2
            squared_errors.append(error)
        
        # RMSE agregado
        if len(squared_errors) == 0:
            return 1e10
        
        rmse = np.sqrt(np.mean(squared_errors))
        return rmse

    def optimize_point_multiyear(self, nc_file, cluster_id, max_params_per_stage=5):
        """
        Otimiza parâmetros usando TODOS os anos simultaneamente.
        
        Fluxo:
        1. Carregar dados de 40 anos do NetCDF
        2. Preparar contexto multi-anual (weather/agromanagement para cada ano)
        3. Executar otimização hierárquica com função objetivo agregada
        4. Salvar parâmetros otimizados + métricas por ano
        
        Args:
            nc_file: Caminho do arquivo NetCDF
            cluster_id: ID do cluster
            max_params_per_stage: Número de parâmetros por etapa hierárquica
            
        Returns:
            dict: Resultados da otimização contendo:
                - optimized_params: Dicionário com parâmetros otimizados
                - rmse_aggregated: RMSE agregado de todos os anos
                - yearly_metrics: DataFrame com métricas por ano
                - execution_time: Tempo de execução (segundos)
        """
        start_time = time.time()
        
        # Carregar dados do ponto
        point_info, weather_df = self.nc_loader.load_point_data(nc_file)
        point_id = point_info['point_id']
        
        print(f"\n{'='*70}")
        print(f"🎯 OTIMIZAÇÃO MULTI-ANUAL - Point {point_id} (Cluster {cluster_id})")
        print(f"{'='*70}")
        
        # Preparar contexto multi-anual
        print("📦 Preparando dados de todos os anos...")
        years_data = self.prepare_multiyear_context(point_info, weather_df, cluster_id)
        
        if len(years_data) == 0:
            print("⚠️  Nenhum ano com dados válidos encontrado")
            return None
        
        print(f"📅 Anos com dados válidos: {len(years_data)}")
        print(f"   Anos: {[y['year'] for y in years_data]}")
        
        # Obter lista ordenada de parâmetros (por importância)
        all_params = self.CLUSTER_PARAMS[cluster_id]
        
        # Dicionário para armazenar valores otimizados acumulados
        optimized_params = {}
        min_rmse = float('inf')
        optimization_stopped_early = False
        
        # Otimização incremental (hierárquica)
        n_stages = (len(all_params) + max_params_per_stage - 1) // max_params_per_stage
        
        print(f"\n🔧 Iniciando otimização hierárquica ({n_stages} etapas)")
        
        for stage in range(1, n_stages + 1):
            # Parâmetros a otimizar nesta etapa
            end_idx = min(stage * max_params_per_stage, len(all_params))
            params_to_optimize = all_params[:end_idx]
            
            new_params = params_to_optimize[max(0, end_idx - max_params_per_stage):end_idx]
            print(f"\n{'─'*70}")
            print(f"📊 Etapa {stage}/{n_stages}: Otimizando {len(params_to_optimize)} parâmetros")
            print(f"   Novos parâmetros: {new_params}")
            
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
            
            # Contexto multi-anual com parâmetros fixos
            context = {
                'years_data': years_data,
                'param_names': params_to_optimize,
                'fixed_params': optimized_params.copy()
            }
            
            opt.set_min_objective(lambda X, grad: self.objective_function_multiyear(X, grad, context))
            
            # Critérios de parada usando constantes
            max_eval_stage = self.max_eval // n_stages
            opt.set_maxeval(max_eval_stage)
            opt.set_xtol_rel(XTOL_REL_EARLY_STAGES if stage <= EARLY_STAGE_THRESHOLD else XTOL_REL_LATER_STAGES)
            opt.set_ftol_abs(FTOL_ABS)  # Tolerância absoluta: 0.1 kg/ha (equivalente a 100 g/ha)
            
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
                stage_start = time.time()
                x_opt = opt.optimize(x0)
                min_rmse = opt.last_optimum_value()
                stage_time = time.time() - stage_start
                
                # Atualizar parâmetros otimizados
                for i, param_name in enumerate(params_to_optimize):
                    optimized_params[param_name] = x_opt[i]
                
                print(f"   ✅ RMSE: {min_rmse:.2f} kg/ha ({stage_time:.1f}s)")
                
                # Condição de parada antecipada usando constante
                if min_rmse < EARLY_STOPPING_RMSE_THRESHOLD:
                    print(f'\n🎉 RMSE < {EARLY_STOPPING_RMSE_THRESHOLD} kg/ha alcançado! Interrompendo otimização.')
                    optimization_stopped_early = True
                    break
                
            except Exception as e:
                print(f"   ❌ Erro na etapa {stage}: {e}")
                break
        
        # Calcular métricas individuais por ano com parâmetros otimizados
        print(f"\n📈 Calculando métricas por ano...")
        yearly_metrics = self._calculate_yearly_metrics(optimized_params, years_data)
        
        # Salvar resultados
        total_time = time.time() - start_time
        results = {
            'point_id': point_id,
            'cluster_id': cluster_id,
            'optimized_params': optimized_params,
            'rmse_aggregated': min_rmse,
            'yearly_metrics': yearly_metrics,
            'n_stages': stage if optimization_stopped_early else n_stages,
            'stopped_early': optimization_stopped_early,
            'execution_time': total_time,
            'n_years': len(years_data)
        }
        
        self.save_multiyear_results(results)
        
        # Limpar arquivos temporários
        for year_data in years_data:
            if os.path.exists(year_data['agro_temp_file']):
                os.remove(year_data['agro_temp_file'])
        
        status = "PERFEITO ✨" if optimization_stopped_early else "Completo"
        print(f"\n{'='*70}")
        print(f"🎉 Otimização {status}!")
        print(f"   RMSE agregado: {min_rmse:.2f} kg/ha")
        print(f"   Tempo total: {total_time:.1f}s")
        print(f"   Anos processados: {len(years_data)}")
        print(f"{'='*70}\n")
        
        return results

    def _calculate_yearly_metrics(self, optimized_params, years_data):
        """
        Calcula métricas individuais para cada ano usando parâmetros otimizados.
        
        Args:
            optimized_params: Dicionário com parâmetros otimizados
            years_data: Lista com dados de cada ano
            
        Returns:
            DataFrame com métricas por ano (year, dyield_obs, dyield_sim, error, rmse, bias, r2)
        """
        metrics_list = []
        
        for year_data in years_data:
            yield_sim = self.run_wofost_simulation(
                optimized_params,
                year_data['parameters'],
                year_data['weather'],
                year_data['agromanagement']
            )
            
            dyield_obs = year_data['dyield_target']
            error = yield_sim - dyield_obs if not np.isnan(yield_sim) else np.nan
            
            metrics_list.append({
                'year': year_data['year'],
                'dyield_obs': dyield_obs,
                'dyield_sim': yield_sim,
                'error': error,
                'abs_error': abs(error) if not np.isnan(error) else np.nan
            })
        
        metrics_df = pd.DataFrame(metrics_list)
        
        # Calcular R² se houver dados suficientes
        if len(metrics_df) > 1 and not metrics_df['dyield_sim'].isna().all():
            valid_data = metrics_df[~metrics_df['dyield_sim'].isna()]
            if len(valid_data) > 1:
                ss_res = np.sum((valid_data['dyield_obs'] - valid_data['dyield_sim']) ** 2)
                ss_tot = np.sum((valid_data['dyield_obs'] - valid_data['dyield_obs'].mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                metrics_df['r2'] = r2
            else:
                metrics_df['r2'] = np.nan
        else:
            metrics_df['r2'] = np.nan
        
        # Calcular bias médio
        if not metrics_df['error'].isna().all():
            metrics_df['bias'] = metrics_df['error'].mean()
        else:
            metrics_df['bias'] = np.nan
        
        return metrics_df

    def save_multiyear_results(self, results):
        """
        Salva resultados da calibração multi-anual em múltiplos formatos.
        
        Arquivos gerados:
        1. OPT_MULTIYEAR_cluster{X}_point{Y}_params.csv:
           - 1 linha com parâmetros otimizados
           
        2. OPT_MULTIYEAR_cluster{X}_point{Y}_yearly_metrics.csv:
           - N linhas (1 por ano) com métricas individuais
           
        3. OPT_MULTIYEAR_cluster{X}_point{Y}_summary.json:
           - Métricas agregadas, tempo de execução, configuração
        
        Args:
            results: Dicionário com resultados da otimização
        """
        point_id = results['point_id']
        cluster_id = results['cluster_id']
        
        base_filename = f"OPT_MULTIYEAR_cluster{cluster_id}_point{point_id}"
        output_dir = self.paths['OPTIMIZATION']
        
        # 1. Salvar parâmetros otimizados
        params_df = pd.DataFrame([results['optimized_params']])
        params_df.insert(0, 'point_id', point_id)
        params_df.insert(1, 'cluster_id', cluster_id)
        params_file = os.path.join(output_dir, f"{base_filename}_params.csv")
        params_df.to_csv(params_file, index=False)
        print(f"   💾 Parâmetros salvos: {params_file}")
        
        # 2. Salvar métricas por ano
        yearly_metrics = results['yearly_metrics'].copy()
        yearly_metrics.insert(0, 'point_id', point_id)
        yearly_metrics.insert(1, 'cluster_id', cluster_id)
        metrics_file = os.path.join(output_dir, f"{base_filename}_yearly_metrics.csv")
        yearly_metrics.to_csv(metrics_file, index=False)
        print(f"   💾 Métricas anuais salvas: {metrics_file}")
        
        # 3. Salvar resumo em JSON
        summary = {
            'point_id': point_id,
            'cluster_id': cluster_id,
            'rmse_aggregated': float(results['rmse_aggregated']),
            'n_years': results['n_years'],
            'n_stages': results['n_stages'],
            'stopped_early': results['stopped_early'],
            'execution_time_seconds': results['execution_time'],
            'mean_abs_error': float(yearly_metrics['abs_error'].mean()) if 'abs_error' in yearly_metrics else None,
            'bias': float(yearly_metrics['bias'].iloc[0]) if 'bias' in yearly_metrics and len(yearly_metrics) > 0 else None,
            'r2': float(yearly_metrics['r2'].iloc[0]) if 'r2' in yearly_metrics and len(yearly_metrics) > 0 else None,
            'algorithm': str(self.algorithm),
            'max_eval': self.max_eval
        }
        
        summary_file = os.path.join(output_dir, f"{base_filename}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   💾 Resumo salvo: {summary_file}")

    def compare_optimization_strategies(self, nc_file, cluster_id):
        """
        Executa e compara calibração anual vs multi-anual.
        
        Compara:
        - Tempo de execução
        - RMSE médio
        - Variabilidade dos parâmetros entre anos
        - Número de avaliações da função objetivo
        
        Args:
            nc_file: Caminho do arquivo NetCDF
            cluster_id: ID do cluster
        
        Returns:
            dict: Relatório comparativo com métricas de ambas as estratégias
        """
        print(f"\n{'='*70}")
        print("📊 COMPARAÇÃO: Calibração Anual vs Multi-Anual")
        print(f"{'='*70}")
        
        # 1. Executar otimização multi-anual
        print("\n🔵 Executando otimização MULTI-ANUAL...")
        multiyear_start = time.time()
        try:
            multiyear_results = self.optimize_point_multiyear(nc_file, cluster_id)
            multiyear_time = time.time() - multiyear_start
        except Exception as e:
            print(f"❌ Erro na otimização multi-anual: {e}")
            return None
        
        # 2. Executar otimização anual (usando metodo herdado)
        print("\n🔴 Executando otimização ANUAL (para comparação)...")
        annual_start = time.time()
        try:
            annual_results = self.optimize_point_hierarchical(nc_file, cluster_id)
            annual_time = time.time() - annual_start
        except Exception as e:
            print(f"❌ Erro na otimização anual: {e}")
            return None
        
        # 3. Comparar resultados com validação robusta
        if not multiyear_results:
            print("❌ Otimização multi-anual falhou")
            return None
            
        if annual_results is None:
            print("❌ Otimização anual falhou")
            return None
        
        # Extrair métricas com tratamento de erro
        try:
            # annual_results é um DataFrame
            if isinstance(annual_results, pd.DataFrame):
                annual_mean_rmse = annual_results['final_error'].mean() if 'final_error' in annual_results.columns else None
                annual_n_years = len(annual_results)
            else:
                annual_mean_rmse = None
                annual_n_years = 0
                
            comparison = {
                'point_id': multiyear_results['point_id'],
                'cluster_id': cluster_id,
                'multiyear': {
                    'execution_time': multiyear_time,
                    'rmse_aggregated': multiyear_results['rmse_aggregated'],
                    'n_years': multiyear_results['n_years']
                },
                'annual': {
                    'execution_time': annual_time,
                    'mean_rmse': annual_mean_rmse,
                    'n_years': annual_n_years
                },
                'speedup': annual_time / multiyear_time if multiyear_time > 0 else None
            }
            
            # Salvar relatório
            point_id = multiyear_results['point_id']
            comparison_file = os.path.join(
                self.paths['OPTIMIZATION'],
                f"comparison_point{point_id}.json"
            )
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            print(f"\n{'='*70}")
            print("📈 RESULTADOS DA COMPARAÇÃO")
            print(f"{'='*70}")
            print(f"⏱️  Tempo Multi-anual: {multiyear_time:.1f}s")
            print(f"⏱️  Tempo Anual: {annual_time:.1f}s")
            if comparison['speedup']:
                print(f"🚀 Speedup: {comparison['speedup']:.1f}x")
            print(f"💾 Relatório salvo: {comparison_file}")
            print(f"{'='*70}\n")
            
            return comparison
            
        except Exception as e:
            print(f"❌ Erro ao processar resultados da comparação: {e}")
            return None

    def cross_validate_temporal(self, nc_file, cluster_id, n_folds=5):
        """
        Validação cruzada temporal para avaliar generalização do modelo.
        
        Estratégia:
        - Dividir anos em n_folds (ex: 40 anos / 5 folds = 8 anos por fold)
        - Para cada fold:
          * Treinar nos outros folds (calibração)
          * Validar no fold atual
        - Calcular métricas agregadas de validação
        
        Args:
            nc_file: Caminho do arquivo NetCDF
            cluster_id: ID do cluster
            n_folds: Número de folds para validação cruzada (default: 5)
            
        Returns:
            dict: Resultados da validação cruzada contendo:
                - fold_results: Lista com resultados de cada fold
                - mean_rmse_validation: RMSE médio de validação
                - std_rmse_validation: Desvio padrão do RMSE de validação
                - params_stability: Variabilidade dos parâmetros entre folds
        """
        print(f"\n{'='*70}")
        print(f"🔬 VALIDAÇÃO CRUZADA TEMPORAL ({n_folds} folds)")
        print(f"{'='*70}")
        
        # Carregar dados
        point_info, weather_df = self.nc_loader.load_point_data(nc_file)
        point_id = point_info['point_id']
        
        # Preparar todos os dados dos anos
        years_data = self.prepare_multiyear_context(point_info, weather_df, cluster_id)
        
        if len(years_data) < n_folds:
            print(f"⚠️  Dados insuficientes para {n_folds} folds (apenas {len(years_data)} anos)")
            return None
        
        # Dividir anos em folds
        years_per_fold = len(years_data) // n_folds
        fold_results = []
        all_params = {}  # Para calcular estabilidade
        
        for fold_idx in range(n_folds):
            print(f"\n{'─'*70}")
            print(f"📁 Fold {fold_idx + 1}/{n_folds}")
            
            # Definir índices de validação
            val_start = fold_idx * years_per_fold
            val_end = val_start + years_per_fold if fold_idx < n_folds - 1 else len(years_data)
            
            # Separar dados de treino e validação
            train_data = years_data[:val_start] + years_data[val_end:]
            val_data = years_data[val_start:val_end]
            
            train_years = [y['year'] for y in train_data]
            val_years = [y['year'] for y in val_data]
            
            print(f"   Treino: {len(train_data)} anos {train_years[:3]}...{train_years[-3:]}")
            print(f"   Validação: {len(val_data)} anos {val_years}")
            
            # Otimizar com dados de treino
            optimized_params, train_rmse = self._optimize_fold(train_data, cluster_id)
            
            # Validar com dados de validação
            val_rmse, val_metrics = self._validate_fold(optimized_params, val_data)
            
            print(f"   ✅ RMSE Treino: {train_rmse:.2f} kg/ha")
            print(f"   📊 RMSE Validação: {val_rmse:.2f} kg/ha")
            
            fold_results.append({
                'fold': fold_idx + 1,
                'train_years': train_years,
                'val_years': val_years,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'optimized_params': optimized_params
            })
            
            # Armazenar parâmetros para análise de estabilidade
            for param, value in optimized_params.items():
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append(value)
        
        # Calcular métricas agregadas
        val_rmses = [f['val_rmse'] for f in fold_results]
        mean_val_rmse = np.mean(val_rmses)
        std_val_rmse = np.std(val_rmses)
        
        # Calcular estabilidade dos parâmetros (coeficiente de variação)
        params_stability = {}
        for param, values in all_params.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = (std_val / mean_val * 100) if mean_val != 0 else 0
            params_stability[param] = {
                'mean': mean_val,
                'std': std_val,
                'cv_percent': cv
            }
        
        # Resultados
        cv_results = {
            'point_id': point_id,
            'cluster_id': cluster_id,
            'n_folds': n_folds,
            'fold_results': fold_results,
            'mean_rmse_validation': mean_val_rmse,
            'std_rmse_validation': std_val_rmse,
            'params_stability': params_stability
        }
        
        # Salvar resultados
        cv_file = os.path.join(
            self.paths['OPTIMIZATION'],
            f"cross_validation_point{point_id}_cluster{cluster_id}.json"
        )
        with open(cv_file, 'w') as f:
            # Converter para formato serializável
            serializable_results = {
                'point_id': point_id,
                'cluster_id': cluster_id,
                'n_folds': n_folds,
                'mean_rmse_validation': float(mean_val_rmse),
                'std_rmse_validation': float(std_val_rmse),
                'validation_rmses': [float(r) for r in val_rmses],
                'params_stability': {
                    k: {sk: float(sv) for sk, sv in v.items()}
                    for k, v in params_stability.items()
                }
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"📈 RESULTADOS DA VALIDAÇÃO CRUZADA")
        print(f"{'='*70}")
        print(f"   RMSE médio de validação: {mean_val_rmse:.2f} ± {std_val_rmse:.2f} kg/ha")
        print(f"   💾 Resultados salvos: {cv_file}")
        print(f"{'='*70}\n")
        
        return cv_results

    def _optimize_fold(self, train_data, cluster_id):
        """
        Otimiza parâmetros usando dados de treino de um fold.
        
        Args:
            train_data: Lista com dados dos anos de treino
            cluster_id: ID do cluster
            
        Returns:
            tuple: (optimized_params dict, train_rmse float)
        """
        all_params = self.CLUSTER_PARAMS[cluster_id]
        
        # Simplificado: usar todos os parâmetros de uma vez para CV
        problem = self.setup_optimization_problem(cluster_id)
        stage_lower = [problem['bounds'][p][0] for p in all_params]
        stage_upper = [problem['bounds'][p][1] for p in all_params]
        
        # Criar otimizador
        opt = nlopt.opt(self.algorithm, len(all_params))
        opt.set_lower_bounds(stage_lower)
        opt.set_upper_bounds(stage_upper)
        
        # Contexto
        context = {
            'years_data': train_data,
            'param_names': all_params,
            'fixed_params': {}
        }
        
        opt.set_min_objective(lambda X, grad: self.objective_function_multiyear(X, grad, context))
        opt.set_maxeval(self.max_eval // 2)  # Reduzir para validação cruzada
        opt.set_xtol_rel(1e-4)
        
        # Ponto inicial (meio do intervalo)
        x0 = [(problem['bounds'][p][0] + problem['bounds'][p][1]) / 2 for p in all_params]
        
        # Otimizar
        x_opt = opt.optimize(x0)
        train_rmse = opt.last_optimum_value()
        
        # Extrair parâmetros otimizados
        optimized_params = {all_params[i]: x_opt[i] for i in range(len(all_params))}
        
        return optimized_params, train_rmse

    def _validate_fold(self, optimized_params, val_data):
        """
        Valida parâmetros otimizados com dados de validação.
        
        Args:
            optimized_params: Dicionário com parâmetros otimizados
            val_data: Lista com dados dos anos de validação
            
        Returns:
            tuple: (val_rmse float, val_metrics DataFrame)
        """
        squared_errors = []
        
        for year_data in val_data:
            yield_sim = self.run_wofost_simulation(
                optimized_params,
                year_data['parameters'],
                year_data['weather'],
                year_data['agromanagement']
            )
            
            if not np.isnan(yield_sim):
                error = (yield_sim - year_data['dyield_target']) ** 2
                squared_errors.append(error)
        
        val_rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else np.nan
        
        # Calcular métricas detalhadas
        val_metrics = self._calculate_yearly_metrics(optimized_params, val_data)
        
        return val_rmse, val_metrics


def main():
    """Função principal para testar otimização multi-anual"""
    start_time = time.time()
    
    # Configurar caminhos
    paths = setup_paths()
    
    # Carregar dados NetCDF
    nc_loader = NetCDFDataLoader(paths['COMPLETO'])
    
    # Criar otimizador multi-anual com configuração padrão recomendada
    optimizer = WOFOSTMultiYearOptimizer(
        paths=paths,
        nc_loader=nc_loader,
        algorithm=nlopt.LN_BOBYQA,  # Algoritmo recomendado (antes: COBYLA)
        max_eval=3000  # Valor padrão recomendado (use 3000 para testes mais rápidos)
    )
    
    # Testar com primeiro arquivo de cada cluster
    print("🧪 Testando otimização multi-anual com 1 ponto por cluster...")
    
    tested_clusters = set()
    for nc_file in nc_loader.nc_files[:10]:  # Limitar a 10 primeiros para teste
        point_info = nc_loader.get_point_info(nc_file)
        cluster_id = point_info['cluster_id']
        
        if cluster_id not in tested_clusters:
            print(f"\n{'='*70}")
            print(f"Testando Cluster {cluster_id}")
            print(f"{'='*70}")
            
            result = optimizer.optimize_point_multiyear(nc_file, cluster_id)
            
            if result:
                print(f"✅ Cluster {cluster_id}: RMSE = {result['rmse_aggregated']:.2f} kg/ha")
                tested_clusters.add(cluster_id)
    
    total_time = time.time() - start_time
    print(f"\n✅ Teste concluído! Tempo total: {total_time:.1f}s")
    print(f"   Resultados salvos em: {paths['OPTIMIZATION']}")


if __name__ == "__main__":
    main()
