import json
import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

import config
from etl import load_full_train_data


def train():
    """Train the model and save it locally"""
    logger = logging.getLogger('train.py')
    logger.info('START - Train model')

    logger.info('Load full data for training')
    X, y = load_full_train_data()
    logger.debug(f'Loaded data has {X.shape[0]} rows and {X.shape[1]} columns')

    params_model = config.get_params_model()
    logger.debug('Parameters of model to validate: \n ' + json.dumps(params_model, indent=2))

    logger.info('Train model')
    model = XGBClassifier(**params_model)
    model.fit(X, y)

    logger.info('Model summary:')
    y_pred_prob = model.predict_proba(X)[:, 1]
    score_ = roc_auc_score(y, y_pred_prob)
    logger.info(f' - AUC (train) = {score_:.3f}')

    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}) \
        .sort_values(['importance'], ascending=False) \
        .reset_index(drop=True)
    logger.info(' - feature importance (top 25): \n'
                + tabulate(feat_imp.head(25), headers=feat_imp.columns, showindex=False) + '\n')

    logger.info(
        f'Save artifacts: \n'
        f' - Model to: {config.FILE_MODEL} \n'
        f' - Features importance to: {config.FILE_MODEL_FEAT_IMP}'
    )
    model.save_model(config.FILE_MODEL)
    feat_imp.to_csv(config.FILE_MODEL_FEAT_IMP, index=False, float_format='%.3f')

    logger.info('Check if model can be loaded back from disk')
    model_loaded = XGBClassifier()
    model_loaded.load_model(config.FILE_MODEL)
    y_pred_check = model_loaded.predict_proba(X)[:, 1]
    assert all(np.array(y_pred_prob - y_pred_check) < 0.00001)

    logger.info('END - Train model')


if __name__ == "__main__":
    train()
