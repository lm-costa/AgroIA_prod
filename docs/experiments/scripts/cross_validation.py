"""
Validacao cruzada da calibracao regional da soja/PR (um conjunto de parametros para o estado).

  CV_MODE=spatial   k particoes de MUNICIPIOS (estratificadas por cluster); treina em ate CV_TRAIN_CAP
                    municipios dos demais e testa nos municipios da particao (todas as safras).
  CV_MODE=temporal  blocos contiguos de SAFRAS; treina nas outras safras e testa nas do bloco,
                    nos mesmos municipios.
  CV_MODE=both      os dois.

Retomavel: particoes concluidas (metrics.json) sao puladas; a particao em andamento retoma do
checkpoint por etapa. Saidas em output/soja_pr/Optimization/cv_<CV_TAG>/.

Variaveis: CV_TAG, CV_MODE, CV_K (5), CV_BLOCKS (3), CV_TRAIN_CAP (50), CV_SEED (42), CV_WORKERS (8),
CV_MAX_EVAL (1500), CV_N_PARAMS (44), CV_YEARS ("2016-2024"), CV_MAX_FOLDS (so para testes de fumaca).
"""
import json, os, sys, time
ROOT = r"D:\_py\AgroIA_prod"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "util")); os.chdir(ROOT)

env = os.environ.get
TAG, MODE = env("CV_TAG", "base"), env("CV_MODE", "both")
K, BLOCKS, TRAIN_CAP, SEED = int(env("CV_K", 5)), int(env("CV_BLOCKS", 3)), int(env("CV_TRAIN_CAP", 50)), int(env("CV_SEED", 42))
WORKERS, MAX_EVAL, N_PARAMS = int(env("CV_WORKERS", 8)), int(env("CV_MAX_EVAL", 1500)), int(env("CV_N_PARAMS", 44))
Y0, Y1 = (int(x) for x in env("CV_YEARS", "2016-2024").split("-"))
MAX_FOLDS = int(env("CV_MAX_FOLDS", 0)) or None


def main():
    import nlopt
    import pandas as pd
    from util.SensitivityAnalyzer_soja_pr import SoyNetCDFDataLoader
    from util.NLOPT_regional_soja_pr import SoyWOFOSTRegionalOptimizerPR, load_overall_ranking, skip_pcse_forced_gc
    from util.utils_soja_pr import setup_paths_soja_pr
    from util.cv_utils_soja_pr import make_spatial_folds, make_year_blocks, skill_metrics, stratified_sample, summarize_folds

    paths = setup_paths_soja_pr()
    out_root = os.path.join(paths['OPTIMIZATION'], f"cv_{TAG}")
    os.makedirs(out_root, exist_ok=True)
    loader = SoyNetCDFDataLoader(paths['COMPLETO'])
    pid = lambda f: os.path.basename(f)[len("point_id_"):-3]
    files_by_cluster, cluster_of = {}, {}
    for f in loader.nc_files:
        c = int(loader.get_point_info(f)['cluster_id'])
        files_by_cluster.setdefault(c, []).append(f)
        cluster_of[pid(f)] = c
    group_params = load_overall_ranking(os.path.join(paths['RESULTS'], 'SA_MORRIS_overall_summary.csv'), top_n=N_PARAMS)
    opt = SoyWOFOSTRegionalOptimizerPR(paths=dict(paths), nc_loader=loader, cluster_params={},
                                       algorithm=nlopt.LN_BOBYQA, max_eval=MAX_EVAL)

    def run_fold(name, train_files, train_years, test_files, test_years):
        fold_dir = os.path.join(out_root, name)
        done = os.path.join(fold_dir, "metrics.json")
        if os.path.exists(done):
            print(f"[{name}] ja concluida, pulando", flush=True)
            return json.load(open(done))
        os.makedirs(fold_dir, exist_ok=True)
        opt.paths['OPTIMIZATION'] = fold_dir
        t0 = time.time()
        res = opt.optimize_group(train_files, 'PR', group_params, n_workers=WORKERS, resume=True,
                                 years=train_years)
        failed = res is None or res.get('calibration_failed', False)
        record = {'fold': name, 'n_train_municipios': len(train_files),
                  'train_years': sorted(train_years) if train_years else 'todas',
                  'test_years': sorted(test_years) if test_years else 'todas',
                  'n_test_municipios': len(test_files), 'calibration_failed': bool(failed),
                  'train_seconds': time.time() - t0}
        if not failed:
            params = res['optimized_params']
            rows = []
            with skip_pcse_forced_gc(True):
                for f in test_files:
                    info, wdf = loader.load_point_data(f)
                    yd = [y for y in opt.prepare_multiyear_context(info, wdf, info['point_id'])
                          if y['dyield_target'] > 0 and (test_years is None or y['year'] in test_years)]
                    if not yd:
                        continue
                    m_cal = opt._calculate_yearly_metrics(params, yd)
                    m_def = opt._calculate_yearly_metrics({}, yd)
                    for y in yd:
                        if os.path.exists(y['agro_temp_file']):
                            os.remove(y['agro_temp_file'])
                    rows.append(pd.DataFrame({'point_id': info['point_id'],
                                              'cluster': cluster_of[str(info['point_id'])],
                                              'year': m_cal['year'], 'obs': m_cal['dyield_obs'],
                                              'sim': m_cal['dyield_sim'], 'sim_padrao': m_def['dyield_sim']}))
            pred = pd.concat(rows, ignore_index=True)
            pred.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)
            record['calibrado'] = skill_metrics(pred)
            record['padrao'] = skill_metrics(pred.assign(sim=pred['sim_padrao']))
            record['por_cluster'] = {str(c): skill_metrics(g) for c, g in pred.groupby('cluster')}
        json.dump(record, open(done, "w"), indent=2)
        shown = ({k: round(v, 3) for k, v in record['calibrado'].items() if k in ('rmse', 'bias', 'r', 'r_temporal')}
                 if not failed else "CALIBRACAO FALHOU")
        print(f"[{name}] feito em {record['train_seconds'] / 60:.0f} min: {shown}", flush=True)
        return record

    results = {}
    if MODE in ("spatial", "both"):
        folds = make_spatial_folds(files_by_cluster, K, SEED)
        json.dump({f"fold{i}": [pid(f) for f in fl] for i, fl in enumerate(folds)},
                  open(os.path.join(out_root, "spatial_partitions.json"), "w"), indent=1)
        recs = []
        for k, test in enumerate(folds[:MAX_FOLDS]):
            held = set(test)
            pool = {c: [f for f in fl if f not in held] for c, fl in files_by_cluster.items()}
            train = stratified_sample(pool, TRAIN_CAP, SEED + k)
            recs.append(run_fold(f"spatial_fold{k}", train, None, test, None))
        results['spatial'] = recs
    if MODE in ("temporal", "both"):
        base = stratified_sample(files_by_cluster, TRAIN_CAP, SEED)
        years = list(range(Y0, Y1 + 1))
        blocks = make_year_blocks(years, BLOCKS)
        json.dump({'municipios': [pid(f) for f in base], 'blocos': blocks},
                  open(os.path.join(out_root, "temporal_partitions.json"), "w"), indent=1)
        recs = []
        for b, block in enumerate(blocks[:MAX_FOLDS]):
            train_years = set(years) - set(block)
            recs.append(run_fold(f"temporal_block{b}", base, train_years, base, set(block)))
        results['temporal'] = recs

    summary = {}
    for kind, recs in results.items():
        ok = [r for r in recs if not r['calibration_failed']]
        summary[kind] = {'partitions': len(recs), 'failed': len(recs) - len(ok)}
        if not ok:
            continue
        for tag in ('calibrado', 'padrao'):
            summary[kind][f'por_particao_{tag}'] = summarize_folds([r[tag] for r in ok])
        preds = pd.concat([pd.read_csv(os.path.join(out_root, r['fold'], "predictions.csv")).assign(fold=r['fold'])
                           for r in ok], ignore_index=True)
        preds.to_csv(os.path.join(out_root, f"{kind}_predictions_all.csv"), index=False)
        summary[kind]['agregado_calibrado'] = skill_metrics(preds)
        summary[kind]['agregado_padrao'] = skill_metrics(preds.assign(sim=preds['sim_padrao']))
        summary[kind]['por_cluster'] = {str(c): skill_metrics(g) for c, g in preds.groupby('cluster')}
    json.dump(summary, open(os.path.join(out_root, "cv_summary.json"), "w"), indent=2)
    print("FIM -- cv_summary.json gravado", flush=True)


if __name__ == "__main__":
    main()
