import pandas as pd
import numpy as np
import json
import logging
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from tabulate import tabulate
import optuna
import copy

import config
from etl import load_train_test_data


X_train, X_test, y_train, y_test = load_train_test_data(selected_feats_only=True)


def objective_xgb(trial):
    """ Objective function to tune an `XGBRegressor` model. """
    params_trial = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.30, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.20, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.01, 5, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 20, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20, log=True),
    }
    params_model = copy.deepcopy(config.PARAMS_MODEL_DEFAULT)
    params_model.update(params_trial)
    model = XGBClassifier(**params_model)
    skf = StratifiedKFold(
        n_splits=config.TUNE_N_SPLITS,
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
    return np.mean(cv['test_score']) #, np.mean(cv['train_score'])


def tune_hyper_params_w_optuna():
    logger = logging.getLogger('tune_hyper_params_optuna.py')
    logger.info('START - Tune hyper-parameters with optuna')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_xgb, n_trials=100)

    metric_names = ['mean_test_score']
    df_cv_results = pd.DataFrame([dict(zip(metric_names, trial.values), **trial.params) for trial in study.trials])
    df_cv_results.sort_values(by=['mean_test_score'], ascending=False, inplace=True)
    logger.info(' - Best parameters found: \n' + json.dumps(study.best_params, indent=2))
    logger.info(' - CV results: \n' + tabulate(df_cv_results.head(25), headers=df_cv_results.columns, showindex=False))

    logger.info(
        f'Save artifacts: \n'
        f' - CV results to: {config.FILE_TUNE_ALL_PARAMS_COMBS} \n'
        f' - Best params to: {config.FILE_TUNE_PARAMS_BEST}'
    )
    config.make_dir_for_artifacts()
    df_cv_results.to_csv(config.FILE_TUNE_ALL_PARAMS_COMBS, index=False, float_format='%.4f')
    with open(config.FILE_TUNE_PARAMS_BEST, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    logger.info('END - Tune hyper-parameters')


def plot_parallel_optuna_res():
    import plotly.offline as py
    import plotly.express as px
    df_res_tune = pd.read_csv(config.FILE_TUNE_ALL_PARAMS_COMBS)
    fig = px.parallel_coordinates(
        df_res_tune,
        color='mean_test_score',
        dimensions=['mean_test_score'] + list(config.TUNE_PARAMS_GRID.keys()),
        color_continuous_scale=['red', 'yellow', 'green'],
        color_continuous_midpoint=np.median(df_res_tune['mean_test_score']),
    )
    # py.iplot(fig)
    py.plot(fig)


if __name__ == "__main__":
    tune_hyper_params_w_optuna()
