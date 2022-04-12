import os
import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier, Booster
from sklearn.metrics import roc_auc_score, accuracy_score
from tabulate import tabulate

from etl.transform import load_raw_data, transform_input_df
from config import DIR_PROJ, TARGET_NAME, get_params_xgboost, get_feat_names

# create dir for artifacts if not existing
if not os.path.exists(f'{DIR_PROJ}/artifacts'):
    os.makedirs(f'{DIR_PROJ}/artifacts')

# set logging
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[logging.FileHandler(f'{DIR_PROJ}/artifacts/logs.log'), logging.StreamHandler()]
)

logging.info('load and prep data')
df = load_raw_data()
df = transform_input_df(df)
df_test = df.loc[df[TARGET_NAME].isna(), ]  # portion of data with unknown target
df = df.loc[~df[TARGET_NAME].isna(), ]

logging.info('train model')
params_xgboost = get_params_xgboost()
feat_names = get_feat_names()
model = XGBClassifier(**params_xgboost)
_ = model.fit(df[feat_names], df[TARGET_NAME])

logging.info('model summary:')
y_pred_prob = pd.DataFrame(model.predict_proba(df[feat_names]))[1]
auc_score_ = roc_auc_score(df[TARGET_NAME], y_pred_prob)
logging.info(f' - AUC (train) = {auc_score_:.3f}')

y_pred = model.predict(df[feat_names])
acc_ = accuracy_score(df[TARGET_NAME], y_pred)
logging.info(f' - accuracy (train) = {acc_:0.3f}')

feat_imp = pd.DataFrame({'feature': feat_names, 'importance': model.feature_importances_})\
    .sort_values(['importance'], ascending=False)\
    .reset_index(drop=True)
logging.info(' - feature importance (top 20): \n'
             + tabulate(feat_imp.head(20), headers=feat_imp.columns, showindex=False))

file_model = f'{DIR_PROJ}/artifacts/model_xgboost.json'
logging.info(f'save model artifact to: {file_model}')
model.save_model(file_model)

logging.info('check if model can be loaded from disk:')
model_loaded = Booster(model_file=file_model)
y_pred_check = model.predict(df[feat_names])
assert all(np.array(y_pred == y_pred_check))
