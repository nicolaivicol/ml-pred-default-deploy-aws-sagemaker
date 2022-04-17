import os
import json
from io import StringIO
import pandas as pd
import numpy as np
from xgboost import XGBClassifier


def model_fn(model_dir):
    """
    Deserialize the pre-trained model
    :param model_dir: SageMaker model directory
    :return: XGBoost model
    """
    model = XGBClassifier()
    model.load_model(os.path.join(model_dir, 'model.json'))
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize the Invoke request body into an object we can perform prediction on
    :param request_body: body of the request sent to the model
    :param request_content_type: (string) specifies the format/variable type of the request
    :return: pd.DataFrame
    """
    if request_content_type == 'application/json':
        return pd.DataFrame(json.loads(request_body))  # expect a table as json
    if request_content_type == 'text/csv':
        return pd.read_csv(StringIO(request_body), header=None)  # read the raw input data as CSV
    else:
        raise ValueError("This model supports the following input types only: application/json, text/csv")


def predict_fn(input_data, model):
    """
    Takes the deserialized request object and performs inference against the loaded model.
    :param input_data: data returned by input_fn
    :param model: model loaded by model_fn
    :return: numpy array
    """
    pred_prob = np.around(model.predict_proba(input_data)[:, 1], decimals=4)
    return pred_prob


# INFO: I am leaving the output_fn as is by default
# https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html#sagemaker-xgboost-model-server
