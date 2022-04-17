# ******************************************************************************
# This contains all configs/parameters used in this project.
# ******************************************************************************

from pathlib import Path
import os
import json
import copy
import logging
from typing import List

# Directories
# ******************************************************************************
DIR_PROJ = (Path(__file__) / '..').resolve()
DIR_DATA = f'{DIR_PROJ}/data'
DIR_ARTIFACTS = f'{DIR_PROJ}/artifacts'

# Data
# ******************************************************************************
FILE_DATA = f'{DIR_DATA}/dataset.csv'
MAP_DATA_COLS_TYPES = {
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

# Data tranformtion & feature engineering
# ******************************************************************************
# categories for simple numerical encoding, integer numbers assigned sequentially for each category
MAP_ENC_CAT = {
    'name_in_email': ['F', 'F+L', 'F1+L', 'Initials', 'L', 'L1+F', 'Nick', 'no_match'],
    'merchant_category': [
        'Diversified entertainment', 'Youthful Shoes & Clothing', 'Books & Magazines',
        'General Shoes & Clothing', 'Concept stores & Miscellaneous', 'Sports gear & Outdoor',
        'Dietary supplements', 'Diversified children products', 'Diversified electronics',
        'Prints & Photos', 'Children Clothes & Nurturing products',
    ],
    'merchant_group': [
        'Entertainment', 'Clothing & Shoes', 'Leisure, Sport & Hobby', 'Health & Beauty',
        'Children Products', 'Home & Garden', 'Electronics', 'Intangible products',
        'Jewelry & Accessories', 'Automotive Products', 'Erotic Materials',
    ]
}

# Target & features
# ******************************************************************************
TARGET_NAME = 'default'
ID_NAME = 'uuid'
FEAT_NAMES_ORIG = [f for f in MAP_DATA_COLS_TYPES.keys() if f not in [TARGET_NAME, ID_NAME]]
FEAT_NAMES_ENG = []  # names of additional engineered features (derived the from initial features)
FEAT_NAMES = FEAT_NAMES_ORIG + FEAT_NAMES_ENG

# Test data
# ******************************************************************************
# Leaving a portion of the data apart for testing.
# This test data is not used in CV, feature selection or hyper parameters tuning.
TEST_SIZE = 0.30
TEST_RANDOM_STATE = 42

# Feature selection
# ******************************************************************************
SLCT_FEATS_MAX_ITER = 100
SLCT_FEATS_N_ESTIMATORS = 200
SLCT_FEATS_LEARNING_RATE = 0.05
SLCT_FEATS_RANDOM_STATE = None
# - artifacts:
FILE_SLCT_FEATS = f'{DIR_ARTIFACTS}/selected_feats.json'
FILE_SLCT_FEATS_RES_BORUTA = f'{DIR_ARTIFACTS}/selected_feats_res_boruta.csv'

# Cross-validation
# ******************************************************************************
CV_N_SPLITS = 16
CV_METRIC = 'roc_auc'
CV_RANDOM_STATE = None
# - artifacts:
FILE_CV_METRICS = f'{DIR_ARTIFACTS}/cv_metrics.csv'
FILE_CV_FEAT_IMP = f'{DIR_ARTIFACTS}/cv_feat_imp.csv'
FILE_TEST_METRICS = f'{DIR_ARTIFACTS}/test_metrics.csv'

# Train model
# ******************************************************************************
# Default parameters for the XGBoost model.
# These can be overriden by the best parameters found via hyper-parameters search.
PARAMS_MODEL_DEFAULT = {
    'objective': 'binary:logistic',
    'use_label_encoder': False,
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'n_estimators': 500,
    'learning_rate': 0.02,
    'max_depth': 4,  # 3,
    'max_leaves': 15,  # 8,
    'colsample_bytree': 0.50,
    'subsample': 0.50,
    'gamma': 1,
    'scale_pos_weight': 1,
    'verbosity': 1,
    'importance_type': 'gain',
    'seed': 42,
    # 'enable_categorical': False,
}
# - artifacts:
FILE_MODEL = f'{DIR_ARTIFACTS}/model.json'
FILE_MODEL_FEAT_IMP = f'{DIR_ARTIFACTS}/model_feat_imp.csv'

# Tuning/search of hyper parameter using randomized grid search & stratified k-fold CV
# ******************************************************************************
TUNE_N_ITER = 250  # number of candidate combinations of parameters to try (aka iterations)
TUNE_N_SPLITS = 4  # number of splits in k-fold validation for each combination of parameters (iteration)
TUNE_SCORING = 'roc_auc'
TUNE_RANDOM_STATE = None  # random each time
# parameters grid to search:
TUNE_PARAMS_GRID = {
    'n_estimators': [200],
    'learning_rate': [0.025, 0.05, 0.10],
    'max_depth': [3, 4, 5],
    # 'max_leaves': 8,  # 15,  # allow for maximum leaves depending on max_depth
    'colsample_bytree': [0.3, 0.5, 0.75, 0.90],
    'subsample': [0.50, 0.70, 0.90],
    'gamma': [0.01, 0.5, 1, 3],
    'min_child_weight': [1, 5, 10],
    'scale_pos_weight': [1, 10, 70, 100],
}
# - artifacts:
FILE_TUNE_PARAMS_BEST = f'{DIR_ARTIFACTS}/tune_params_best.json'
FILE_TUNE_ALL_PARAMS_COMBS = f'{DIR_ARTIFACTS}/tune_all_params_combs.csv'

# Prediction
# ******************************************************************************
PRED_TARGET_NAME = 'pd'
PRED_ID_NAME = 'uuid'
FILE_DATA_TO_PREDICT = f'{DIR_DATA}/dataset.csv'
# - artifact:
FILE_PRED = f'{DIR_ARTIFACTS}/pred-[TIMESTAMP].csv'

# Deployment
# ******************************************************************************
# https://us-east-1.console.aws.amazon.com/iamv2/home?region=us-east-1#/roles
AWS_SAGEMAKER_ROLE_ARN = 'arn:aws:iam::323062068303:role/service-role/AmazonSageMaker-ExecutionRole-20220411T215390'
AWS_SAGEMAKER_ENDPOINT_MODEL = 'klrn-model-xgb-ep-2022-04-17-20-59-14'

# Logging
# ******************************************************************************
LOGS_LEVEL = logging.DEBUG
FILE_LOGS = f'{DIR_ARTIFACTS}/logs.log'
# set logging config:
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(FILE_LOGS), logging.StreamHandler()],
    level=LOGS_LEVEL,
)


def get_params(default_params, file_json_new_params) -> dict:
    """
    Return default parameters overriden by new parameters from json file.
    :param default_params: default parameters
    :param file_json_new_params: file with new parameters
    :return: dictionary with parameters and values
    """
    params = copy.deepcopy(default_params)
    if os.path.exists(file_json_new_params):
        with open(file_json_new_params, 'r') as f:
            new_params = json.load(f)
            params.update(new_params)  # override
    return params


def get_params_model() -> dict:
    """
    Get parameters of the model. \n
    By default, return default parameters from config. But if tuning was
    performed, then override default parameters with best parameters from
    the file generated by the tuning step.
    :return: dictionary with parameters and values
    """
    params = get_params(PARAMS_MODEL_DEFAULT, FILE_TUNE_PARAMS_BEST)
    return params


def get_feat_names() -> List[str]:
    """
    Get all feature names to be used by the model. \n
    By default, return all features. But if feature selection was performed,
    load the list of features from the file generated by the feature selection step.
    :return: list of features
    """
    feat_names = FEAT_NAMES.copy()
    if os.path.exists(FILE_SLCT_FEATS):
        with open(FILE_SLCT_FEATS, 'r') as f:
            feat_names = json.load(f)
    return feat_names


def make_dir_for_artifacts():
    """Create dir for artifacts if it does not exist"""
    if not os.path.exists(DIR_ARTIFACTS):
        os.makedirs(DIR_ARTIFACTS)
