import os
import json
import pandas as pd
import logging
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier
from tabulate import tabulate
import time

from etl import load_transform_data
from config import TARGET_NAME, get_feat_names, PARAMS_XGBOOST_DEFAULT, \
    FILE_CV_RES_BEST_PARAMS_SEARCH, FILE_PARAMS_XGBOOST_BEST_BY_CV, \
    DIR_ARTIFACTS, N_ITER, TEST_SIZE, N_SPLITS, RANDOM_STATE, PARAMS_GRID

# create dir for artifacts if not existing
if not os.path.exists(DIR_ARTIFACTS):
    os.makedirs(DIR_ARTIFACTS)

logger = logging.getLogger('tune_hyper_params')
logger.info('START tune_hyper_params.py')
logger.info(
    f'Params for hyper parameter tuning: \n'
    f'N_ITER: {N_ITER} \n'
    f'TEST_SIZE: {TEST_SIZE} \n'
    f'N_SPLITS: {N_SPLITS} \n'
    f'RANDOM_STATE: {RANDOM_STATE} \n'
    f'PARAMS_GRID: \n'
    f'{json.dumps(PARAMS_GRID, indent=2)}'
)

logger.info('Load and prepare data')
df = load_transform_data()
X_train, X_test, y_train, y_test = train_test_split(
    df[get_feat_names()],
    df[TARGET_NAME],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

logger.info('Search of parameters')
tic = time.time()
model_xgb = XGBClassifier(**PARAMS_XGBOOST_DEFAULT)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
rnd_search = RandomizedSearchCV(
    estimator=model_xgb,
    param_distributions=PARAMS_GRID,
    n_iter=N_ITER,
    scoring='roc_auc',
    n_jobs=-1,
    cv=skf.split(X_train, y_train),
    verbose=10,
    random_state=RANDOM_STATE,
)
rnd_search.fit(X_train, y_train)
logger.info(f'Search of parameters completed. Time elapsed: {((time.time() - tic) / 60):0.2f} minutes')

logger.info(
    f'Metrics:\n'
    f' - AUC score (CV): {rnd_search.best_score_:0.3f} \n'
    f' - AUC score (train): {rnd_search.score(X_train, y_train):0.3f} \n'
    f' - AUC score (test): {rnd_search.score(X_test, y_test):0.3f} \n'
)

df_cv_results = pd.DataFrame(rnd_search.cv_results_).sort_values(by=['rank_test_score']).reset_index(drop=True)
cols_show = [f'param_{p}' for p in PARAMS_GRID.keys()] + ['mean_test_score', 'std_test_score', 'mean_fit_time']
logger.info(' - CV results: \n' + tabulate(df_cv_results[cols_show].head(25), headers=cols_show, showindex=False))
logger.info(' - Best parameters found: \n' + json.dumps(rnd_search.best_params_, indent=2))

logger.info(
    f'Save artifacts: \n'
    f' - CV results to: {FILE_CV_RES_BEST_PARAMS_SEARCH} \n'
    f' - Best params to: {FILE_PARAMS_XGBOOST_BEST_BY_CV}'
)
df_cv_results.to_csv(FILE_CV_RES_BEST_PARAMS_SEARCH, index=False, float_format='%.3f')

with open(FILE_PARAMS_XGBOOST_BEST_BY_CV, 'w') as fp:
    json.dump(rnd_search.best_params_, fp)

logger.info('END tune_hyper_params.py')
