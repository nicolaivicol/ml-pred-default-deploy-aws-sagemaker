import json
import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier
from boruta import BorutaPy
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

import config
from etl import load_train_test_data


def select_feats():
    """Perform feature selection with Boruta"""
    logger = logging.getLogger('select_feats.py')
    logger.info('START - Feature selection')

    logger.info('Load data')
    X_train, X_test, y_train, y_test = load_train_test_data(selected_feats_only=False)
    feat_names = X_train.columns.to_list()

    params_model = config.get_params_model()
    params_model['n_estimators'] = config.SLCT_FEATS_N_ESTIMATORS
    params_model['learning_rate'] = config.SLCT_FEATS_LEARNING_RATE
    logger.debug('Model parameters: \n ' + json.dumps(params_model, indent=2))

    logger.info(f'Run feature selection ({len(feat_names)} features)')
    model = XGBClassifier(**params_model)
    fs = BorutaPy(
        estimator=model,
        n_estimators='auto',
        max_iter=config.SLCT_FEATS_MAX_ITER,
        verbose=10,
        random_state=config.SLCT_FEATS_RANDOM_STATE,
    )
    fs.fit(X_train.values, y_train.values)

    confirmed_feats = X_train.columns[fs.support_].to_list()
    tentative_feats = X_train.columns[fs.support_weak_].to_list()
    selected_feats = confirmed_feats + tentative_feats  # select both accepted and undecided
    dropped_feats = [f for f in feat_names if f not in selected_feats]

    df_res = pd.DataFrame({'feature': feat_names, 'ranking': fs.ranking_, 'support': np.repeat('', len(feat_names))})
    df_res.loc[np.array([i in confirmed_feats for i in feat_names]), 'support'] = 'Confirmed'
    df_res.loc[[i in tentative_feats for i in feat_names], 'support'] = 'Tentative'
    df_res.loc[[i not in selected_feats for i in feat_names], 'support'] = 'Rejected'
    df_res = df_res.sort_values(by='ranking').reset_index(drop=True)
    logger.info(f'Features selected: {len(selected_feats)}, dropped: {len(dropped_feats)}')
    logger.info('Features, ranking & support: \n' + tabulate(df_res, df_res.columns, showindex=False) + '\n')

    model.fit(X_train[selected_feats], y_train)
    auc_score_train = roc_auc_score(y_train, model.predict_proba(X_train[selected_feats])[:, 1])
    auc_score_test = roc_auc_score(y_test, model.predict_proba(X_test[selected_feats])[:, 1])
    logger.info(
        f'Metrics of model with selected features: \n'
        f' - AUC (train) = {auc_score_train:.3f} \n'
        f' - AUC (test) = {auc_score_test:.3f}'
    )

    logger.info(
        f'Save artifacts: \n'
        f' - Boruta results to: {config.FILE_SLCT_FEATS_RES_BORUTA} \n'
        f' - Selected features to: {config.FILE_SLCT_FEATS}'
    )
    config.make_dir_for_artifacts()
    df_res.to_csv(config.FILE_SLCT_FEATS_RES_BORUTA, index=False, float_format='%.3f')
    with open(config.FILE_SLCT_FEATS, 'w') as f:
        json.dump(selected_feats, f, indent=2)

    logger.info('END - Feature selection')


if __name__ == "__main__":
    select_feats()
