import boto3
import json
import unittest
import numpy as np
import logging
from xgboost import XGBClassifier

import config
from etl import load_full_train_data

logger = logging.getLogger('test_invoke_aws_sagemaker.py')


class TestInvokeAwsSagemaker(unittest.TestCase):

    def setUp(self) -> None:
        logger.info('Load data')
        X, y = load_full_train_data()
        self.X_sample = X.head(100)

        logger.info('Load model from disk and predict for a small sample')
        model_loaded = XGBClassifier()
        model_loaded.load_model(config.FILE_MODEL)
        self.pred_y_local = model_loaded.predict_proba(self.X_sample)[:, 1]

        self.runtime_client = boto3.client('sagemaker-runtime')

    def test_inference_input_json(self):
        response = self.runtime_client.invoke_endpoint(
            EndpointName=config.AWS_SAGEMAKER_ENDPOINT_MODEL,
            ContentType='application/json',
            Body=self.X_sample.to_json(orient='values'),
        )
        pred_y_aws = json.loads(response['Body'].read().decode())
        assert all(np.array(self.pred_y_local - pred_y_aws) < 0.001)

    def test_inference_input_csv(self):
        response = self.runtime_client.invoke_endpoint(
            EndpointName=config.AWS_SAGEMAKER_ENDPOINT_MODEL,
            ContentType='text/csv',
            Body=self.X_sample.to_csv(index=False, header=False),
        )
        pred_y_aws = json.loads(response['Body'].read().decode())
        assert all(np.array(self.pred_y_local - pred_y_aws) < 0.001)
