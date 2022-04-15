import json
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

import config
from etl import load_train_test_data


def cv():
    """Run cross-validation with best parameters to estimate model performance out-of-sample"""
    logger = logging.getLogger('cv.py')
    logger.info('START - Cross-validation')

    logger.info('Load data')
    X_train, X_test, y_train, y_test = load_train_test_data(selected_feats_only=True)

    params_model = config.get_params_model()
    logger.debug('Parameters of model to validate: \n' + json.dumps(params_model, indent=2))

    logger.info('Run cross-validation')
    model = XGBClassifier(**params_model)
    skf = StratifiedKFold(
        n_splits=config.CV_N_SPLITS,
        shuffle=True,
        random_state=config.CV_RANDOM_STATE,
    )
    cv = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        scoring=config.CV_METRIC,
        cv=skf.split(X_train, y_train),
        n_jobs=-1,
        verbose=10,
        return_estimator=True,
        return_train_score=True,
    )

    # collect features importance from all iterations
    l_feat_imp = []

    for i, estimator in enumerate(cv['estimator']):
        df_feat_imp_i = pd.DataFrame({
            'fold': np.repeat(i + 1, len(X_train.columns)),
            'feature': X_train.columns,
            'importance': estimator.feature_importances_})
        l_feat_imp.append(df_feat_imp_i)

    df_feat_imp = pd.concat(l_feat_imp)
    df_scores = pd.DataFrame({'train_score': cv['train_score'], 'test_score': cv['test_score']})

    # train model on train data and evaluate on test data (the portion not involved in CV)
    model = XGBClassifier(**params_model)
    model.fit(X_train, y_train)
    y_pred_prob_test = model.predict_proba(X_test)[:, 1]
    score_test = roc_auc_score(y_test, y_pred_prob_test)

    avg_train, std_train = np.mean(cv['train_score']), np.std(cv['train_score'])
    avg_test, std_test = np.mean(cv['test_score']), np.std(cv['test_score'])
    metric_ = config.CV_METRIC.upper()
    logger.info(
        f'Metrics: \n'
        f' - train: avg {metric_} = {avg_train:.3f} (std {metric_} = {std_train:.3f})\n'
        f' - valid: avg {metric_} = {avg_test:.3f} (std {metric_} = {std_test:.3f})\n'
        f' - test: {metric_} = {score_test:.3f}'
    )

    logger.info(
        f'Save artifacts: \n'
        f' - Metrics to: {config.FILE_CV_METRICS} \n'
        f' - Features importance to: {config.FILE_CV_FEAT_IMP}'
    )
    config.make_dir_for_artifacts()
    df_scores.to_csv(config.FILE_CV_METRICS, index=False, float_format='%.3f')
    df_feat_imp.to_csv(config.FILE_CV_FEAT_IMP, index=False, float_format='%.3f')
    pd.DataFrame({'test_score': [score_test]}).to_csv(config.FILE_TEST_METRICS, index=False, float_format='%.3f')

    logger.info('END - Cross-validation')


if __name__ == "__main__":
    cv()
