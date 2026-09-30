"""
Calibracao REGIONAL do WOFOST para soja/PR: um unico conjunto de parametros
por grupo (um cluster inteiro, ou o Parana inteiro), otimizado contra o RMSE
agregado de todos os municipios x safras do grupo -- em vez de um conjunto de
parametros por municipio (util/NLOPT_soja_pr.py), pensado para inferencia
posterior em qualquer ponto do grupo em vez de ficar preso a serie curta
(10 anos) de um unico municipio.

Reaproveita a maior parte de SoyWOFOSTMultiYearOptimizerPR (montagem do clima
e do agromanagement por municipio/ano via prepare_multiyear_context) -- so a
orquestracao da otimizacao e a funcao objetivo sao diferentes, porque:

1) a funcao objetivo passa a agregar centenas/milhares de amostras
   (municipio x ano) em vez de ~10, entao usa uma tolerancia a falhas
   (FAILURE_TOLERANCE) em vez de descartar a tentativa inteira por causa de
   uma unica simulacao invalida (objective_function_multiyear original faz
   isso, o que travaria a otimizacao regional quase sempre);
2) o resultado salvo e por GRUPO (cluster ou estado), nao por municipio.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from NLOPT_soja_pr import SoyWOFOSTMultiYearOptimizerPR, EXCLUDED_PARAMS
from utils import WOFOST_bounds

EARLY_STOPPING_RMSE_THRESHOLD = 100  # kg/ha -- mesmo criterio da calibracao por municipio
XTOL_REL_EARLY_STAGES = 1e-5
XTOL_REL_LATER_STAGES = 1e-4
FTOL_ABS = 0.1
EARLY_STAGE_THRESHOLD = 3

# Fracao maxima de amostras (municipio x ano) que podem falhar em uma
# avaliacao antes de penalizar a tentativa inteira com 1e10. Com centenas de
# amostras agregadas, exigir 100% de sucesso (como no caso por municipio)
# travaria a otimizacao quase sempre.
FAILURE_TOLERANCE = 0.2

# Produtividade simulada abaixo deste piso (kg/ha) e tratada como falha,
# igual a uma simulacao que retornou NaN. Sem esse piso, um conjunto de
# parametros que faz a lavoura "falhar sempre" (TWSO ~ 0, um resultado
# numericamente valido, nao NaN) pode passar despercebido por
# FAILURE_TOLERANCE e ainda minimizar o RMSE se a amostra de calibracao
# tiver, por acaso, produtividade media baixa -- um ajuste que nao reflete
# fenologia real e nao generaliza (e exatamente o padrao observado na
# validacao holdout: parametros de um grupo simulando ~0 para qualquer
# municipio de fora da amostra, independente do clima real daquele ano).
MIN_PLAUSIBLE_YIELD = 300.0

# Se uma etapa "convergir" com RMSE perto do teto de penalidade (1e10,
# devolvido por objective_function_regional quando falhas demais acontecem),
# isso NAO significa que o NLOPT achou um bom ajuste -- so significa que toda
# a regiao testada nessa etapa falhou igual, e o 1e10 foi aceito como se
# fosse um minimo valido (nenhuma excecao e lancada nesse caso, entao o
# codigo aceitava esses parametros sem questionar). Qualquer RMSE acima
# deste teto e tratado como etapa nao convergida -- essa e a causa raiz do
# colapso observado no cluster 1.0 (parametros da etapa 8/9 travados no
# 1e10 e aceitos, produzindo produtividade simulada ~0 para qualquer
# municipio na validacao holdout).
FAILURE_CEILING = 1e9


class SoyWOFOSTRegionalOptimizerPR(SoyWOFOSTMultiYearOptimizerPR):
    """Calibracao regional (1 conjunto de parametros por cluster ou por estado inteiro)."""

    def prepare_regional_samples(self, nc_files):
        """
        Junta os anos de TODOS os municipios de `nc_files` em uma unica lista
        de amostras, reaproveitando prepare_multiyear_context por municipio.

        Usa o point_id (em vez do cluster_id) como "sal" do nome do arquivo
        temporario de agromanagement, para nunca colidir entre municipios do
        mesmo cluster/ano (prepare_multiyear_context nao inclui o point_id
        no nome do arquivo, so o cluster_id -- inofensivo quando chamado uma
        vez por municipio, mas aqui chamamos varias vezes em sequencia).
        """
        samples = []
        for nc_file in nc_files:
            point_info, weather_df = self.nc_loader.load_point_data(nc_file)
            years_data = self.prepare_multiyear_context(point_info, weather_df, point_info['point_id'])
            for yd in years_data:
                yd['point_id'] = point_info['point_id']
            samples.extend(years_data)
        return samples

    def objective_function_regional(self, X, grad, context):
        """
        RMSE agregado sobre todas as amostras do grupo, tolerando ate
        FAILURE_TOLERANCE de falhas de simulacao (em vez de descartar a
        tentativa inteira, como objective_function_multiyear faz).
        """
        model_params = self.extract_model_params(X, context['param_names'])
        model_params.update(context.get('fixed_params', {}))

        squared_errors = []
        n_falhas = 0

        for sample in context['years_data']:
            yield_sim = self.run_wofost_simulation(
                model_params, sample['parameters'], sample['weather'], sample['agromanagement']
            )
            if np.isnan(yield_sim) or yield_sim < MIN_PLAUSIBLE_YIELD:
                n_falhas += 1
                continue
            squared_errors.append((yield_sim - sample['dyield_target']) ** 2)

        n_total = len(context['years_data'])
        if n_total == 0 or len(squared_errors) == 0 or (n_falhas / n_total) > FAILURE_TOLERANCE:
            return 1e10

        return np.sqrt(np.mean(squared_errors))

    def _checkpoint_path(self, group_label):
        return os.path.join(self.paths['OPTIMIZATION'], f"OPT_REGIONAL_group{group_label}_checkpoint.json")

    def _save_checkpoint(self, group_label, stage_completed, optimized_params, min_rmse, group_params, max_params_per_stage):
        checkpoint = {
            'group_label': group_label,
            'stage_completed': stage_completed,
            'optimized_params': {k: float(v) for k, v in optimized_params.items()},
            'min_rmse': float(min_rmse),
            'group_params': group_params,
            'max_params_per_stage': max_params_per_stage,
        }
        tmp_path = self._checkpoint_path(group_label) + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        os.replace(tmp_path, self._checkpoint_path(group_label))

    def _load_checkpoint(self, group_label, group_params, max_params_per_stage):
        path = self._checkpoint_path(group_label)
        if not os.path.exists(path):
            return None

        with open(path) as f:
            checkpoint = json.load(f)

        if checkpoint.get('group_params') != group_params or checkpoint.get('max_params_per_stage') != max_params_per_stage:
            print(f"   ⚠️  Checkpoint de '{group_label}' encontrado mas com parametros/config diferentes -- ignorando e recomecando do zero.")
            return None

        return checkpoint

    def optimize_group(self, nc_files, group_label, group_params, max_params_per_stage=5, resume=True):
        """
        Otimizacao hierarquica (mesmo esquema em etapas de
        optimize_point_multiyear), mas sobre a lista combinada de amostras
        de varios municipios, produzindo UM conjunto de parametros para o
        grupo inteiro.

        Salva um checkpoint (OPT_REGIONAL_group{label}_checkpoint.json) a
        cada etapa concluida -- cada etapa pode levar horas com centenas de
        amostras agregadas, entao perder tudo numa interrupcao (queda de
        energia, kernel reiniciado, etc.) e caro demais para nao ter como
        retomar. Com resume=True (padrao), se existir um checkpoint
        compativel (mesmos group_params e max_params_per_stage), a
        otimizacao continua da proxima etapa em vez de recomecar do zero;
        resume=False ignora qualquer checkpoint existente.
        """
        start_time = time.time()

        print(f"\n{'=' * 70}\n🌎 OTIMIZACAO REGIONAL -- Grupo '{group_label}' ({len(nc_files)} municipios)\n{'=' * 70}")

        print("📦 Preparando amostras de todos os municipios do grupo...")
        samples = self.prepare_regional_samples(nc_files)

        if len(samples) == 0:
            print("⚠️  Nenhuma amostra valida encontrada")
            return None

        print(f"📅 Amostras (municipio x safra) validas: {len(samples)}")

        group_params = [p for p in group_params if p not in EXCLUDED_PARAMS]

        n_stages = max(1, (len(group_params) + max_params_per_stage - 1) // max_params_per_stage)

        checkpoint = self._load_checkpoint(group_label, group_params, max_params_per_stage) if resume else None

        if checkpoint is not None:
            optimized_params = checkpoint['optimized_params']
            min_rmse = checkpoint['min_rmse']
            start_stage = checkpoint['stage_completed'] + 1
            print(f"   ♻️  Retomando do checkpoint: etapa {checkpoint['stage_completed']} ja concluida (RMSE = {min_rmse:.2f} kg/ha).")
        else:
            optimized_params = {}
            min_rmse = float('inf')
            start_stage = 1

        optimization_stopped_early = False
        stage = start_stage - 1

        if start_stage > n_stages:
            print(f"   ♻️  Checkpoint ja cobre todas as {n_stages} etapas -- pulando direto para as metricas.")

        print(f"\n🔧 Iniciando/retomando otimizacao hierarquica ({n_stages} etapas, {len(group_params)} parametros)")

        for stage in range(start_stage, n_stages + 1):
            end_idx = min(stage * max_params_per_stage, len(group_params))
            params_to_optimize = group_params[:end_idx]

            bounds = WOFOST_bounds(params_to_optimize)
            stage_lower = [bounds[p][0] for p in params_to_optimize]
            stage_upper = [bounds[p][1] for p in params_to_optimize]

            import nlopt
            opt = nlopt.opt(self.algorithm, len(params_to_optimize))
            opt.set_lower_bounds(stage_lower)
            opt.set_upper_bounds(stage_upper)

            context = {
                'years_data': samples,
                'param_names': params_to_optimize,
                'fixed_params': optimized_params.copy(),
            }

            # Rastreia o melhor (x, RMSE) visto durante a etapa por fora do
            # proprio NLOPT: quando opt.optimize() lanca RoundoffLimited, o
            # NLOPT aborta sem devolver o x correspondente ao ultimo
            # last_optimum_value(), entao sem esse rastreamento perderiamos o
            # progresso da etapa inteira. Tambem e usado para decidir se a
            # etapa bateu no FAILURE_CEILING (nenhuma avaliacao valida).
            best_track = {'x': None, 'rmse': float('inf')}

            def tracked_objective(X, grad, _context=context, _best=best_track):
                rmse = self.objective_function_regional(X, grad, _context)
                if rmse < _best['rmse']:
                    _best['rmse'] = rmse
                    _best['x'] = list(X)
                return rmse

            opt.set_min_objective(tracked_objective)

            max_eval_stage = max(1, self.max_eval // n_stages)
            opt.set_maxeval(max_eval_stage)
            opt.set_xtol_rel(XTOL_REL_EARLY_STAGES if stage <= EARLY_STAGE_THRESHOLD else XTOL_REL_LATER_STAGES)
            opt.set_ftol_abs(FTOL_ABS)

            x0 = []
            for p in params_to_optimize:
                if p in optimized_params:
                    x0.append(optimized_params[p])
                else:
                    lb, ub = bounds[p]
                    x0.append((lb + ub) / 2)

            stage_start = time.time()
            try:
                opt.optimize(x0)
            except nlopt.RoundoffLimited:
                # Codigo -3 do NLOPT ("erro de arredondamento"): a busca
                # estagnou por limite de precisao numerica nesta etapa, nao e
                # uma falha real. Ao contrario de outras excecoes, nao
                # abandona as etapas seguintes -- usa o melhor resultado
                # parcial rastreado em best_track e segue em frente.
                print(f"   ⚠️  Etapa {stage}/{n_stages}: erro de arredondamento (busca estagnada) -- usando melhor resultado parcial.")
            except Exception as e:
                print(f"   ❌ Erro na etapa {stage}: {e}")
                break
            stage_time = time.time() - stage_start

            stage_rmse = best_track['rmse']
            stage_x = best_track['x']

            # Nenhuma avaliacao valida na etapa (ou todas bateram o teto de
            # falha objective_function_regional=1e10): NAO aceita esses
            # parametros, mantem os da etapa anterior e avanca para a
            # proxima etapa (mais parametros podem ajudar a escapar da
            # regiao inviavel). Aceitar silenciosamente aqui e o bug que
            # colapsou o cluster 1.0 -- ver FAILURE_CEILING acima.
            if stage_x is None or stage_rmse >= FAILURE_CEILING:
                print(f"   ⚠️  Etapa {stage}/{n_stages}: nenhum resultado abaixo do teto de falha (RMSE={stage_rmse:.2e}) -- descartando etapa, mantendo parametros anteriores ({stage_time:.1f}s).")
                self._save_checkpoint(group_label, stage, optimized_params, min_rmse, group_params, max_params_per_stage)
                continue

            for i, p in enumerate(params_to_optimize):
                optimized_params[p] = stage_x[i]
            min_rmse = stage_rmse

            print(f"   ✅ Etapa {stage}/{n_stages}: RMSE regional = {min_rmse:.2f} kg/ha ({stage_time:.1f}s)")

            self._save_checkpoint(group_label, stage, optimized_params, min_rmse, group_params, max_params_per_stage)

            if min_rmse < EARLY_STOPPING_RMSE_THRESHOLD:
                print(f"\n🎉 RMSE regional < {EARLY_STOPPING_RMSE_THRESHOLD} kg/ha! Interrompendo otimizacao.")
                optimization_stopped_early = True
                break

        print("\n📈 Calculando metricas por municipio/ano com os parametros do grupo...")
        yearly_metrics = self._calculate_yearly_metrics(optimized_params, samples)
        yearly_metrics.insert(0, 'point_id', [s['point_id'] for s in samples])

        total_time = time.time() - start_time
        result = {
            'group_label': group_label,
            'n_municipios': len(nc_files),
            'optimized_params': optimized_params,
            'rmse_aggregated': min_rmse,
            'yearly_metrics': yearly_metrics,
            'n_stages': stage if optimization_stopped_early else n_stages,
            'stopped_early': optimization_stopped_early,
            'execution_time': total_time,
            'n_samples': len(samples),
        }

        self.save_group_results(result)

        # Resultado final consolidado e salvo -- o checkpoint intermediario
        # nao e mais necessario.
        checkpoint_path = self._checkpoint_path(group_label)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        for sample in samples:
            if os.path.exists(sample['agro_temp_file']):
                os.remove(sample['agro_temp_file'])

        status = "PERFEITO ✨" if optimization_stopped_early else "Completo"
        print(f"\n{'=' * 70}")
        print(f"🎉 Grupo '{group_label}' {status}! RMSE regional: {min_rmse:.2f} kg/ha | Tempo: {total_time:.1f}s")
        print(f"{'=' * 70}\n")

        return result

    def save_group_results(self, result):
        group_label = result['group_label']
        output_dir = self.paths['OPTIMIZATION']
        base_filename = f"OPT_REGIONAL_group{group_label}"

        params_df = pd.DataFrame([result['optimized_params']])
        params_df.insert(0, 'group_label', group_label)
        params_df.insert(1, 'n_municipios', result['n_municipios'])
        params_file = os.path.join(output_dir, f"{base_filename}_params.csv")
        params_df.to_csv(params_file, index=False)

        metrics_file = os.path.join(output_dir, f"{base_filename}_yearly_metrics.csv")
        result['yearly_metrics'].to_csv(metrics_file, index=False)

        summary = {
            'group_label': group_label,
            'n_municipios': result['n_municipios'],
            'n_samples': result['n_samples'],
            'rmse_aggregated': float(result['rmse_aggregated']),
            'n_stages': result['n_stages'],
            'stopped_early': result['stopped_early'],
            'execution_time_seconds': result['execution_time'],
        }
        summary_file = os.path.join(output_dir, f"{base_filename}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"   💾 Parametros: {params_file}")
        print(f"   💾 Metricas anuais: {metrics_file}")
        print(f"   💾 Resumo: {summary_file}")


def load_overall_ranking(overall_csv_path, top_n=44):
    """
    Le o ranking de sensibilidade GERAL (todos os clusters combinados),
    salvo por MorrisScreeningAnalyzer.analyze_overall_results como
    'SA_MORRIS_overall_summary.csv' (indice = parameter, colunas mu_star/sigma/mu).
    Usado para o modo de agregacao 'estado' (nao ha ranking por-cluster que
    faca sentido quando o grupo cruza todos os clusters).
    """
    df = pd.read_csv(overall_csv_path)

    param_col = df.columns[0]
    if param_col != 'parameter':
        df = df.rename(columns={param_col: 'parameter'})

    df = df[~df['parameter'].isin(EXCLUDED_PARAMS)]
    df = df.sort_values('mu_star', ascending=False)

    return df['parameter'].tolist()[:top_n]
