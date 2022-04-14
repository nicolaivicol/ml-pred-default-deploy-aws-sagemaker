# this has utils to load, transform data prior to feed to the model
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

import config
from config import MAP_DATA_COLS_TYPES, MAP_ENC_CAT, FILE_DATA, TARGET_NAME, ID_NAME

cols_numeric = [k for k, v in MAP_DATA_COLS_TYPES.items() if v == 'numeric']
cols_boolean = [k for k, v in MAP_DATA_COLS_TYPES.items() if v in ['boolean']]
cols_categorical = [k for k, v in MAP_DATA_COLS_TYPES.items() if v in ['categorical', 'boolean']]
cols_categorical_text = ['merchant_category', 'merchant_group', 'name_in_email']
cols_categorical_int = [c for c in cols_categorical if c not in (cols_categorical_text + cols_boolean)]


def load_raw_data(file_data=None) -> pd.DataFrame:
    """
    Load raw data from csv file
    :param file_data: path to the csv file with raw data
    :return: data frame
    """
    if file_data is None:
        file_data = FILE_DATA
    df = pd.read_csv(file_data, sep=';', na_values=['NA'])
    return df


def transform_input_df(df, enable_categorical=False) -> pd.DataFrame:
    """
    Transform the input data frame into a data frame with numeric columns only,
    ready to be provided for train.
    :param df: input data frame
    :param enable_categorical: convert categorical to 'category' type, default: False
    :return: data frame
    """

    # sub NA values in numeric columns with -1
    for col in cols_numeric:
        df.loc[df[col].isna(), col] = -1
        df[col] = df[col].astype(float)

    # convert boolean columns to numeric and sub NA values with -1
    for col in cols_boolean:
        df.loc[df[col].isna(), col] = -1
        df[col] = df[col].astype(int)

    # convert categorical columns with int values to int and sub NA values with 0
    for col in cols_categorical_int:
        if col == TARGET_NAME:
            continue
        df.loc[df[col].isna(), col] = 0
        df[col] = df[col].astype(int)

    # convert text categorical columns to int - numeric encoding
    for col in cols_categorical_text:
        v = np.repeat(0, len(df[col]))
        for i in range(len(MAP_ENC_CAT[col])):
            v[df[col] == MAP_ENC_CAT[col][i]] = i + 1
        df[col] = v

    # convert categorical to 'category' type
    if enable_categorical:
        for col in cols_categorical:
            df[col] = df[col].astype('category')

    return df


def load_transform_data(file_data=None, filter_na_target=False, enable_categorical=False) -> pd.DataFrame:
    """
    Run load_raw_data(), transform_input_df() then
    :param file_data: path to the csv file with raw data
    :param filter_na_target: filter rows having NA values for target?, default: False
    :param enable_categorical: convert categorical to 'category' type, default: False
    :return: data frame
    """
    df = load_raw_data(file_data)
    df = transform_input_df(df, enable_categorical=enable_categorical)
    if TARGET_NAME in df.columns.to_list():
        if filter_na_target:
            df = df.loc[df[TARGET_NAME].isna(),]  # portion of data with unknown target which we want to predict
        else:
            df = df.loc[~df[TARGET_NAME].isna(),]
    return df


def load_train_test_data(selected_feats_only=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.array, pd.array]:
    """
    Load train/test data
    :param selected_feats_only: load data only with selected features? Default: False
    :return: X_train, X_test, y_train, y_test
    """
    df = load_transform_data()
    feat_names = config.get_feat_names() if selected_feats_only else config.FEAT_NAMES.copy()
    X_train, X_test, y_train, y_test = train_test_split(
        df[feat_names],
        df[config.TARGET_NAME],
        test_size=config.TEST_SIZE,
        random_state=config.TEST_RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


def load_full_train_data() -> Tuple[pd.DataFrame, pd.array]:
    """
    Load full train data
    :return: X_train, y_train
    """
    df = load_transform_data()
    feat_names = config.get_feat_names()
    X_train, y_train = df[feat_names], df[config.TARGET_NAME]
    return X_train, y_train


def load_predict_data() -> Tuple[pd.DataFrame, pd.array]:
    """
    Load data for prediction
    :return: data frame
    """
    df = load_transform_data(filter_na_target=True)
    feat_names = config.get_feat_names()
    return df[feat_names], df[ID_NAME]
