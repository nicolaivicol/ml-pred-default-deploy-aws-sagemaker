import os
import json
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from etl import load_transform_data
from config import TARGET_NAME, get_feat_names, get_params_xgboost, DIR_ARTIFACTS, FILE_CV_RES

# create dir for artifacts if not existing
if not os.path.exists(DIR_ARTIFACTS):
    os.makedirs(DIR_ARTIFACTS)

logger = logging.getLogger('cv')
logger.info('START cv.py')

logger.info('Load and prepare data')
df = load_transform_data()
X_train, y_train = df[get_feat_names()], df[TARGET_NAME]

logger.info('Run cross-validation')
params_xgboost = get_params_xgboost()
model_xgb = XGBClassifier(**params_xgboost)
skf = StratifiedKFold(n_splits=10, shuffle=True)
scores = cross_val_score(
    estimator=model_xgb,
    X=X_train,
    y=y_train,
    scoring='roc_auc',
    n_jobs=-1,
    cv=skf.split(X_train, y_train),
    verbose=10,
)
avg_auc, std_auc = np.mean(scores), np.std(scores)
logger.info(
    f'Metrics: \n'
    f' - avg AUC (valid) = {avg_auc:.3f} \n'
    f' - std AUC (valid) = {std_auc:.3f}'
)

with open(FILE_CV_RES, 'w') as fp:
    json.dump({'avg AUC': np.mean(scores), 'std AUC': np.std(scores)}, fp, indent=2)

logger.info('END cv.py')
