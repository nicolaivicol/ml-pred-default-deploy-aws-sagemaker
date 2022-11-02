import json
import pandas as pd
import logging
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from tabulate import tabulate
import time

import config
from etl import load_train_test_data


def tune_hyper_params():
    """Search for best hyper-parameters via randomized grid search using stratified k-fold validation"""
    logger = logging.getLogger('tune_hyper_params_rand.py')
    logger.info('START - Tune hyper-parameters')
    logger.debug(
        f'Parameters: \n'
        f' - N_ITER: {config.TUNE_N_ITER} \n'
        f' - N_SPLITS: {config.TUNE_N_SPLITS} \n'
        f' - PARAMS_GRID: \n'
        f'{json.dumps(config.TUNE_PARAMS_GRID, indent=2)}'
    )

    logger.info('Load data')
    X_train, X_test, y_train, y_test = load_train_test_data()

    logger.info('Search of parameters')
    tic = time.time()
    model_xgb = XGBClassifier(**config.PARAMS_MODEL_DEFAULT)
    skf = StratifiedKFold(n_splits=config.TUNE_N_SPLITS, shuffle=True, random_state=config.TUNE_RANDOM_STATE)
    search_cv = RandomizedSearchCV(
        estimator=model_xgb,
        param_distributions=config.TUNE_PARAMS_GRID,
        n_iter=config.TUNE_N_ITER,
        scoring=config.TUNE_SCORING,
        n_jobs=-1,
        cv=skf.split(X_train, y_train),
        verbose=10,
        random_state=config.TUNE_RANDOM_STATE,
    )
    search_cv.fit(X_train, y_train)
    logger.info(f'Search of parameters completed. Time elapsed: {((time.time() - tic) / 60):0.2f} minutes')

    logger.info(
        f'Metrics:\n'
        f' - AUC score (best found in CV): {search_cv.best_score_:0.3f} \n'
        f' - AUC score (train): {search_cv.score(X_train, y_train):0.3f} \n'
        f' - AUC score (test): {search_cv.score(X_test, y_test):0.3f} \n'
    )

    df_cv_results = pd.DataFrame(search_cv.cv_results_).sort_values(by=['rank_test_score']).reset_index(drop=True)
    cols_show = [f'param_{p}' for p in config.TUNE_PARAMS_GRID.keys()] + ['mean_test_score', 'std_test_score', 'mean_fit_time']
    logger.info(' - CV results: \n' + tabulate(df_cv_results[cols_show].head(25), headers=cols_show, showindex=False))
    logger.info(' - Best parameters found: \n' + json.dumps(search_cv.best_params_, indent=2))

    logger.info(
        f'Save artifacts: \n'
        f' - CV results to: {config.FILE_TUNE_ALL_PARAMS_COMBS} \n'
        f' - Best params to: {config.FILE_TUNE_PARAMS_BEST}'
    )
    config.make_dir_for_artifacts()
    df_cv_results.to_csv(config.FILE_TUNE_ALL_PARAMS_COMBS, index=False, float_format='%.3f')
    with open(config.FILE_TUNE_PARAMS_BEST, 'w') as f:
        json.dump(search_cv.best_params_, f, indent=2)

    logger.info('END - Tune hyper-parameters')


if __name__ == "__main__":
    tune_hyper_params()
