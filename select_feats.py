import os
import json
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from boruta import BorutaPy
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

from etl import load_transform_data
from config import TARGET_NAME, PARAMS_XGBOOST_DEFAULT, FEAT_NAMES, \
    FILE_SLCT_FEATS, FILE_RES_SLCT_FEATS_BORUTA, \
    DIR_ARTIFACTS, TEST_SIZE, RANDOM_STATE

# create dir for artifacts if not existing
if not os.path.exists(DIR_ARTIFACTS):
    os.makedirs(DIR_ARTIFACTS)

logger = logging.getLogger('select_feats')
logger.info('START select_feats.py')

logger.info('Load and prepare data')
df = load_transform_data()
feat_names = FEAT_NAMES.copy()
X_train, X_test, y_train, y_test = train_test_split(
    df[feat_names],
    df[TARGET_NAME],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

pars = PARAMS_XGBOOST_DEFAULT.copy()
pars['n_estimators'], pars['learning_rate'] = 200, 0.05  # for faster iterations
model = XGBClassifier(**pars)

logger.info('Run feature selection with boruta')
fs = BorutaPy(estimator=model, n_estimators='auto', max_iter=100, verbose=10, random_state=42)
fs.fit(X_train.values, y_train.values)

confirmed_feats = X_train.columns[fs.support_].to_list()
tentative_feats = X_train.columns[fs.support_weak_].to_list()
selected_feats = confirmed_feats + tentative_feats  # select both accepted and undecided

df_res = pd.DataFrame({'feature': feat_names, 'ranking': fs.ranking_, 'support': np.repeat('', len(feat_names))})
df_res.loc[np.array([i in confirmed_feats for i in feat_names]), 'support'] = 'Confirmed'
df_res.loc[[i in tentative_feats for i in feat_names], 'support'] = 'Tentative'
df_res.loc[[i not in selected_feats for i in feat_names], 'support'] = 'Rejected'
df_res = df_res.sort_values(by='ranking').reset_index(drop=True)
logger.info('Features, ranking & support: \n' + tabulate(df_res, df_res.columns, showindex=False) + '\n')

_ = model.fit(X_train[selected_feats], y_train)
auc_score_train = roc_auc_score(y_train, model.predict_proba(X_train[selected_feats])[:, 1])
auc_score_test = roc_auc_score(y_test, model.predict_proba(X_test[selected_feats])[:, 1])
logger.info(
    f'Metrics of model with selected features: \n'
    f' - AUC (train) = {auc_score_train:.3f} \n'
    f' - AUC (test) = {auc_score_test:.3f}'
)

logger.info(
    f'Save artifacts: \n'
    f' - Boruta results to: {FILE_RES_SLCT_FEATS_BORUTA} \n'
    f' - Selected features to: {FILE_SLCT_FEATS}'
)
df_res.to_csv(FILE_RES_SLCT_FEATS_BORUTA, index=False, float_format='%.3f')

with open(FILE_SLCT_FEATS, 'w') as fp:
    json.dump(selected_feats, fp, indent=2)

logger.info('END select_feats.py')
