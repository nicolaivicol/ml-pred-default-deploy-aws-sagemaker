# This contains all configs/parameters used in this project.

from pathlib import Path
import os
import json
import copy
import logging

DIR_PROJ = (Path(__file__) / '..').resolve()
DIR_ARTIFACTS = f'{DIR_PROJ}/artifacts'

FILE_DATA = f'{DIR_PROJ}/data/dataset.csv'
FILE_PARAMS_XGBOOST_BEST_BY_CV = f'{DIR_ARTIFACTS}/params_xgboost_best_by_cv.json'
FILE_SELECTED_FEATS_BY_CV = f'{DIR_ARTIFACTS}/selected_feats_by_cv.json'
FILE_LOGS = f'{DIR_ARTIFACTS}/logs.log'
FILE_MODEL_XGBOOST = f'{DIR_ARTIFACTS}/model_xgboost.json'
FILE_CV_RES_BEST_PARAMS_SEARCH = f'{DIR_ARTIFACTS}/cv_res_best_params_search.csv'

MAP_DATA_TYPES = {
    'uuid': 'text',
    'default': 'categorical',
    'account_amount_added_12_24m': 'numeric',
    'account_days_in_dc_12_24m': 'numeric',
    'account_days_in_rem_12_24m': 'numeric',
    'account_days_in_term_12_24m': 'numeric',
    'account_incoming_debt_vs_paid_0_24m': 'numeric',
    'account_status': 'categorical',
    'account_worst_status_0_3m': 'categorical',
    'account_worst_status_12_24m': 'categorical',
    'account_worst_status_3_6m': 'categorical',
    'account_worst_status_6_12m': 'categorical',
    'age': 'numeric',
    'avg_payment_span_0_12m': 'numeric',
    'avg_payment_span_0_3m': 'numeric',
    'merchant_category': 'categorical',
    'merchant_group': 'categorical',
    'has_paid': 'boolean',
    'max_paid_inv_0_12m': 'numeric',
    'max_paid_inv_0_24m': 'numeric',
    'name_in_email': 'categorical',
    'num_active_div_by_paid_inv_0_12m': 'numeric',
    'num_active_inv': 'numeric',
    'num_arch_dc_0_12m': 'numeric',
    'num_arch_dc_12_24m': 'numeric',
    'num_arch_ok_0_12m': 'numeric',
    'num_arch_ok_12_24m': 'numeric',
    'num_arch_rem_0_12m': 'numeric',
    'num_arch_written_off_0_12m': 'numeric',
    'num_arch_written_off_12_24m': 'numeric',
    'num_unpaid_bills': 'numeric',
    'status_last_archived_0_24m': 'categorical',
    'status_2nd_last_archived_0_24m': 'categorical',
    'status_3rd_last_archived_0_24m': 'categorical',
    'status_max_archived_0_6_months': 'categorical',
    'status_max_archived_0_12_months': 'categorical',
    'status_max_archived_0_24_months': 'categorical',
    'recovery_debt': 'numeric',
    'sum_capital_paid_account_0_12m': 'numeric',
    'sum_capital_paid_account_12_24m': 'numeric',
    'sum_paid_inv_0_12m': 'numeric',
    'time_hours': 'numeric',
    'worst_status_active_inv': 'categorical',
}

TARGET_NAME = 'default'
FEAT_NAMES_ORIG = [feat for feat in MAP_DATA_TYPES.keys() if feat not in [TARGET_NAME, 'uuid']]
FEAT_NAMES_ENG = []  # names of additional engineered features derived from initial features
FEAT_NAMES = FEAT_NAMES_ORIG + FEAT_NAMES_ENG

MAP_ENC_CAT = {
    'name_in_email': ['F', 'F+L', 'F1+L', 'Initials', 'L', 'L1+F', 'Nick', 'no_match'],
    'merchant_category': [
        'Diversified entertainment', 'Youthful Shoes & Clothing', 'Books & Magazines',
        'General Shoes & Clothing', 'Concept stores & Miscellaneous', 'Sports gear & Outdoor',
    ],
    'merchant_group': [
        'Entertainment', 'Clothing & Shoes', 'Leisure, Sport & Hobby', 'Health & Beauty',
        'Children Products', 'Home & Garden', 'Electronics', 'Intangible products',
    ]
}

# Default parameters for the XGBoost model
PARAMS_XGBOOST_DEFAULT = {
    # booster:
    "objective": "binary:logistic",
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "booster": "gbtree",
    "n_estimators": 500,
    "learning_rate": 0.10,
    "max_depth": 3,  # 4,
    "max_leaves": 8,  # 15,
    "colsample_bytree": 0.70,
    "subsample": 0.70,
    "gamma": 0.01,
    "verbosity": 1,
    "seed": 42,
    # "enable_categorical": False,
    # "nthread": 1,
}

# Parameters for hyper parameter tuning using randomized grid search and stratified k-fold CV
N_ITER = 200  # number of candidate combinations of parameters to try (aka iterations)
TEST_SIZE = 0.30  # portion of data left as test (not used in CV)
N_SPLITS = 4  # number of splits in k-fold validation for each combination of parameters (iteration)
RANDOM_STATE = None  # random each time
# parameters grid to search
PARAMS_GRID = {
    'n_estimators': [200],
    'learning_rate': [0.025, 0.05, 0.10],
    'max_depth': [3, 4, 5],
    # "max_leaves": 8,  # 15,
    'colsample_bytree': [0.3, 0.5, 0.75, 0.90],
    'subsample': [0.50, 0.70, 0.90],
    'gamma': [0.01, 0.5, 1, 3],
    'min_child_weight': [1, 5, 10],
}

# set logging configs
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(FILE_LOGS), logging.StreamHandler()],
    level=logging.DEBUG,
)


def get_params(default_params, file_json_new_params):
    params = copy.deepcopy(default_params)
    if os.path.exists(file_json_new_params):
        with open(file_json_new_params, 'r') as f:
            new_params = json.load(f)
            params.update(new_params)  # override
    return params


def get_params_xgboost():
    params = get_params(PARAMS_XGBOOST_DEFAULT, FILE_PARAMS_XGBOOST_BEST_BY_CV)
    return params


def get_feat_names():
    feat_names = FEAT_NAMES.copy()
    if os.path.exists(FILE_SELECTED_FEATS_BY_CV):
        with open(FILE_SELECTED_FEATS_BY_CV, 'r') as f:
            feat_names = json.load(f)
    return feat_names
