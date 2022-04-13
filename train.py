import json
import os
import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier, Booster
from sklearn.metrics import roc_auc_score, accuracy_score
from tabulate import tabulate

from etl.transform import load_raw_data, transform_input_df
from config import DIR_ARTIFACTS, FILE_MODEL_XGBOOST, TARGET_NAME, get_params_xgboost, get_feat_names

# create dir for artifacts if not existing
if not os.path.exists(DIR_ARTIFACTS):
    os.makedirs(DIR_ARTIFACTS)

# set logging
logger = logging.getLogger('train.py')
logger.info('START train.py')

logger.info('load and prep data')
df = load_raw_data()
df = transform_input_df(df)
df_test = df.loc[df[TARGET_NAME].isna(), ]  # portion of data with unknown target
df = df.loc[~df[TARGET_NAME].isna(), ]
logger.info(f'data loaded: rows: {df.shape[0]} | columns: {df.shape[1]}')

logger.info('load train configs')
params_xgboost = get_params_xgboost()
logger.info('- parameters: \n' + json.dumps(params_xgboost, indent=2))
feat_names = get_feat_names()
logger.info('- features: \n' + json.dumps(feat_names, indent=2))

logger.info('train XGBoost model')
model = XGBClassifier(**params_xgboost)
_ = model.fit(df[feat_names], df[TARGET_NAME])

logger.info('model summary:')
y_pred_prob = pd.DataFrame(model.predict_proba(df[feat_names]))[1]
auc_score_ = roc_auc_score(df[TARGET_NAME], y_pred_prob)
logger.info(f' - AUC (train) = {auc_score_:.3f}')

y_pred = model.predict(df[feat_names])
acc_ = accuracy_score(df[TARGET_NAME], y_pred)
logger.info(f' - accuracy (train) = {acc_:0.3f}')

feat_imp = pd.DataFrame({'feature': feat_names, 'importance': model.feature_importances_})\
    .sort_values(['importance'], ascending=False)\
    .reset_index(drop=True)
logger.info(' - feature importance (top 20): \n'
             + tabulate(feat_imp.head(20), headers=feat_imp.columns, showindex=False))

logger.info(f'save model artifact to disk: {FILE_MODEL_XGBOOST}')
model.save_model(FILE_MODEL_XGBOOST)

logger.info('check if model can be loaded from disk:')
model_loaded = Booster(model_file=FILE_MODEL_XGBOOST)
y_pred_check = model.predict(df[feat_names])
assert all(np.array(y_pred == y_pred_check))

logger.info('END train.py')
