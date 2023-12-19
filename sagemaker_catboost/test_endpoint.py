import boto3
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import sagemaker

import config

# logging.basicConfig(level=logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('s3transfer').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger('sagemaker_catboost/test_endpoint.py')
logger.info('START - sagemaker_catboost/test_endpoint.py')

client = boto3.client(service_name="sagemaker")
# runtime = boto3.client(service_name="sagemaker-runtime")
boto_session = boto3.session.Session()
# s3 = boto_session.resource('s3')
# region = boto_session.region_name
# sagemaker_session = sagemaker.Session()

aws_role = config.AWS_SAGEMAKER_ROLE_ARN
aws_region = boto3.Session().region_name
sess = sagemaker.Session()


logger.info('Test the deployed model')
# **********************************************************************************************************************
content_type = 'text/csv'
endpoint_name = 'default-klrn-catboost-2023-12-08-21-10-34-859'
test_data_bucket = 'misc-datasets-nvicol'
test_data_prefix = 'dataset-default-klrn/test'
test_data_file_name = 'data.csv'

# misc-datasets-nvicol/dataset-default-klrn/test/data.csv

# download data from s3
boto3.client("s3").download_file(
    test_data_bucket,
    f"{test_data_prefix}/{test_data_file_name}",
    test_data_file_name
)

# read the data
test_data = pd.read_csv(test_data_file_name, header=None)
test_data.columns = ["Target"] + [f"Feature_{i}" for i in range(1, test_data.shape[1])]
test_data = test_data.head(1000)

num_examples, num_columns = test_data.shape
logger.info(f"The test dataset contains {num_examples} examples and {num_columns} columns.")

# prepare the ground truth target and predicting features to send into the endpoint.
ground_truth_label, features = test_data.iloc[:, :1], test_data.iloc[:, 1:]
logger.info(f"The first 5 observations of the data: \n" + test_data.head(5).to_string())

# query endpoint

def query_endpoint(encoded_tabular_data):
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Body=encoded_tabular_data
    )
    return response


def parse_response(query_response):
    model_predictions = json.loads(query_response["Body"].read())
    predicted_probabilities = model_predictions["probabilities"]
    return np.array(predicted_probabilities)


# split the test data into smaller size of batches to query the endpoint due to the large size of test data.
batch_size = 1000
predict_prob = []
for i in np.arange(0, num_examples, step=batch_size):
    query_response_batch = query_endpoint(
        features.iloc[i : (i + batch_size), :].to_csv(header=False, index=False).encode("utf-8")
    )
    predict_prob_batch = parse_response(query_response_batch)  # prediction probability per batch
    predict_prob.append(predict_prob_batch)

predict_prob = np.concatenate(predict_prob, axis=0)
predict_prob = np.around(predict_prob[:, 1], decimals=4)
# predict_prob = pd.Series(predict_prob[:, 1]).reset_index(drop=True).to_frame('predict_proba')

th_prob = 0.05
predict_label = predict_prob > th_prob

# Visualize the predictions results by plotting the confusion matrix.
conf_matrix = confusion_matrix(y_true=ground_truth_label.values, y_pred=predict_label)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large")

plt.xlabel("Predictions", fontsize=18)
plt.ylabel("Actuals", fontsize=18)
plt.title("Confusion Matrix", fontsize=18)
plt.show()

# Measure the prediction results quantitatively.
eval_accuracy = accuracy_score(ground_truth_label.values, predict_label)
eval_f1 = f1_score(ground_truth_label.values, predict_label)

print(
    f"Evaluation result on test data: \n"
    f"{accuracy_score.__name__}: {eval_accuracy}\n"
    f"F1 Score: {eval_f1}\n"
)
