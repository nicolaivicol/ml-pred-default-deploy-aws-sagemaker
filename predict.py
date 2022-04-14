import pandas as pd
import logging
from xgboost import XGBClassifier
from tabulate import tabulate
from datetime import datetime

import config
from etl import load_predict_data


def predict():
    logger = logging.getLogger('predict.py')
    logger.info('START - Predict')

    logger.info('Load data for prediction')
    X, ids = load_predict_data()

    logger.info('Load trained model from file')
    model_loaded = XGBClassifier()
    model_loaded.load_model(config.FILE_MODEL)

    df_pred = pd.DataFrame({config.ID_NAME: ids, 'pd': model_loaded.predict_proba(X)[:, 1]})
    logger.info('Sample of prediction: \n' + tabulate(df_pred.head(25), headers=df_pred.columns, showindex=False) + '\n')

    file_pred = config.FILE_PRED.replace('[TIMESTAMP]', datetime.today().strftime('%Y%m%d%H%M%S'))
    logger.info(f'Save prediction as artifact to: {file_pred} \n')
    df_pred.to_csv(file_pred, sep=';', index=False, float_format='%.4f')

    logger.info('END - Predict')


if __name__ == "__main__":
    predict()
