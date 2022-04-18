import boto3
import json
import unittest
import numpy as np
import logging
from xgboost import XGBClassifier
import requests

import config
from etl import load_full_train_data

logger = logging.getLogger('test_api.py')


class TestInvokeAwsSagemaker(unittest.TestCase):

    def setUp(self) -> None:
        logger.info('Load data')
        X, y = load_full_train_data()
        self.X_sample = X.head(100)

        logger.info('Load model from disk and predict for a small sample')
        model_loaded = XGBClassifier()
        model_loaded.load_model(config.FILE_MODEL)
        self.pred_y_local = model_loaded.predict_proba(self.X_sample)[:, 1]

    def test_api(self):
        res = requests.post(
            url=config.AWS_API_URL_RESOURCE,
            data=json.dumps({'data': self.X_sample.to_csv(index=False, header=False)}),
        )
        pred_y_aws = json.loads(res.content)
        assert all(np.array(self.pred_y_local - pred_y_aws) < 0.001)
