# this has utils to load, transform data prior to feed to the model
import os
import boto3
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

import config
from config import MAP_DATA_COLS_TYPES, MAP_ENC_CAT, FILE_DATA, TARGET_NAME, ID_NAME

import json

cols_numeric = [k for k, v in MAP_DATA_COLS_TYPES.items() if v == 'numeric']
cols_boolean = [k for k, v in MAP_DATA_COLS_TYPES.items() if v in ['boolean']]
cols_categorical = [k for k, v in MAP_DATA_COLS_TYPES.items() if v in ['categorical', 'boolean']]
cols_categorical_text = ['merchant_category', 'merchant_group', 'name_in_email']
cols_categorical_int = [c for c in cols_categorical if c not in (cols_categorical_text + cols_boolean)]


def extract_raw_data_from_db():
    """
    Extract data from data base or s3 and save to disk.
    TODO: implement
    :return:
    """
    pass


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


def upload_directory_to_s3(dir_local, s3_bucket, s3_folder):
    """
    Uploads a directory and its subdirectories to an S3 bucket.

    Args:
    local_directory (str): The local directory to upload.
    bucket (str): The S3 bucket to upload to.
    s3_folder (str): The S3 folder path to upload the files to.
    """
    # Create a boto3 client
    s3_client = boto3.client('s3')

    # Walk through the local directory
    for root, dirs, files in os.walk(dir_local):
        for filename in files:
            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full S3 path
            relative_path = os.path.relpath(local_path, dir_local)
            s3_path = os.path.join(s3_folder, relative_path)

            # Upload the file
            s3_client.upload_file(local_path, s3_bucket, s3_path)
            print(f"File {local_path} uploaded to {s3_path}")


def split_raw_data_to_train_test_pred_for_s3_catboost(test_size=0.1, random_state=42,
                                                      folder_name='data_for_s3_catboost'):
    """ split raw data to train/test/pred """
    df = load_raw_data()

    # keep only the target and feature columns, the target columns is first
    df = df[[config.TARGET_NAME] + config.FEAT_NAMES]

    # convert categorical columns with int values to int and sub NA values with 0
    for col in cols_categorical_int:
        is_na = df[col].isna()
        df.loc[is_na, col] = 0
        df[col] = df[col].astype(int)

    # data for prediction with unknown target
    df_pred = df.loc[df[config.TARGET_NAME].isnull(),]

    # train / test split wth known target
    df = df.loc[~df[config.TARGET_NAME].isnull(),]
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    dfs = {
        'train': df_train,
        'test': df_test,
        'pred': df_pred,
    }

    # keep data on disk
    for name, df in dfs.items():
        dir_ = f'{config.DIR_DATA}/{folder_name}/{name}'
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        # save data to .csv
        df.to_csv(f'{dir_}/data.csv', sep=',', index=False, header=False)

        # save the json file with categorical column indexes, this JSON file should be formatted such that the key is
        # 'cat_index' and value is a list of categorical column index.
        dict_ = {'cat_index': [i for i, col in enumerate(df.columns)
                               if col in cols_categorical and col != config.TARGET_NAME]}

        # save to json file
        with open(f'{dir_}/cat_index.json', 'w') as f:
            json.dump(dict_, f)


if __name__ == '__main__':
    split_raw_data_to_train_test_pred_for_s3_catboost()
    upload_directory_to_s3(f'{config.DIR_DATA}/data_for_s3_catboost',
                           'misc-datasets-nvicol',
                           'dataset-default-klrn')
    print('')

