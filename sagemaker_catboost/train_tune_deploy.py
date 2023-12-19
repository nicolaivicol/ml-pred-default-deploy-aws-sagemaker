import os
import boto3
import json
import logging
import time
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sagemaker
from sagemaker import image_uris, model_uris, script_uris, get_execution_role, hyperparameters
from sagemaker.tuner import ContinuousParameter, IntegerParameter, HyperparameterTuner
from sagemaker.estimator import Estimator
from sagemaker.utils import name_from_base
from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig

import config

# logging.basicConfig(level=logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('s3transfer').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger('sagemaker_catboost/train_tune_deploy.py')
logger.info('START - sagemaker_catboost/train_tune_deploy.py')

client = boto3.client(service_name="sagemaker")
# runtime = boto3.client(service_name="sagemaker-runtime")
boto_session = boto3.session.Session()
# s3 = boto_session.resource('s3')
# region = boto_session.region_name
# sagemaker_session = sagemaker.Session()

aws_role = config.AWS_SAGEMAKER_ROLE_ARN
aws_region = boto3.Session().region_name
sess = sagemaker.Session()

# https://docs.aws.amazon.com/sagemaker/latest/dg/catboost.html
# https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/lightgbm_catboost_tabular/Amazon_Tabular_Classification_LightGBM_CatBoost.ipynb


logger.info('Retrieve training artifacts (docker image, train script, etc.)')
# **********************************************************************************************************************
model_name = 'default-klrn-catboost'
train_model_id = 'catboost-classification-model'
train_model_version = '*'
train_scope = 'training'
training_instance_type = 'ml.m5.large'
logger.info(f'model id: {train_model_id}')

logger.info('Retrieve the docker image for training')
train_image_uri = image_uris.retrieve(
    region=None,
    framework=None,
    model_id=train_model_id,
    model_version=train_model_version,
    image_scope=train_scope,
    instance_type=training_instance_type,
)
logger.info(f'Docker image uri: {train_image_uri}')

logger.info('Retrieve the training script')
train_source_uri = script_uris.retrieve(
    model_id=train_model_id,
    model_version=train_model_version,
    script_scope=train_scope
)
logger.info(f'Training script uri: {train_source_uri}')

logger.info('Retrieve the pre-trained model tarball to further fine-tune')
train_model_uri = model_uris.retrieve(
    model_id=train_model_id,
    model_version=train_model_version,
    model_scope=train_scope
)
logger.info(f'pre-trained model tarball uri: {train_model_uri}')


logger.info('Set training parameters')
# **********************************************************************************************************************
# location of training data in s3
training_data_bucket = 'misc-datasets-nvicol'
training_data_prefix = 'dataset-default-klrn/train'
training_dataset_s3_path = f's3://{training_data_bucket}/{training_data_prefix}'

# output location in s3
output_bucket = sess.default_bucket()
output_prefix = "default-klrn"
s3_output_location = f"s3://{output_bucket}/{output_prefix}/output"

# retrieve the default hyperparameters for fine-tuning the model
hyperparameters = hyperparameters.retrieve_default(
    model_id=train_model_id,
    model_version=train_model_version
)

# [Optional] override default hyperparameters with custom values
hyperparameters["eval_metric"] = "AUC"
hyperparameters["iterations"] = "500"
hyperparameters["learning_rate"] = "0.10"
hyperparameters["depth"] = "5"
hyperparameters["l2_leaf_reg"] = "6"
hyperparameters["random_strength"] = "0.15"

logger.info(f'Initial hyperparameters: \n {json.dumps(hyperparameters, indent=2)}')

# define the hyperparameters ranges (search space)
# https://docs.aws.amazon.com/sagemaker/latest/dg/catboost-hyperparameters.html
use_amt = False  # use automatic model tuning?
hyperparameter_ranges = {
    "learning_rate": ContinuousParameter(0.00001, 0.1, scaling_type="Logarithmic"),
    # "iterations": IntegerParameter(50, 1000),
    "depth": IntegerParameter(1, 10),
    "l2_leaf_reg": IntegerParameter(1, 10),
    "random_strength": ContinuousParameter(0.01, 10, scaling_type="Logarithmic"),
}

logger.info('Start training')
# **********************************************************************************************************************
training_job_name = name_from_base(model_name, max_length=80)

# Create SageMaker Estimator instance
estimator = Estimator(
    role=aws_role,
    image_uri=train_image_uri,
    source_dir=train_source_uri,
    model_uri=train_model_uri,
    entry_point="transfer_learning.py",
    instance_count=1,
    instance_type=training_instance_type,
    max_run=1 * 60 * 60,  # seconds
    hyperparameters=hyperparameters,
    output_path=s3_output_location,
)

if use_amt:
    logger.info(f'Start hyperparameters tuning job: {training_job_name} using data from {training_dataset_s3_path}')
    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name='AUC',
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=[{"Name": 'AUC', "Regex": "AUC: ([0-9\\.]+)"}],
        max_jobs=15,
        max_parallel_jobs=2,
        objective_type="Maximize",
        base_tuning_job_name=training_job_name,
    )
    tuner.fit({"training": training_dataset_s3_path}, logs=True)
    tuner.wait()
    estimator = tuner.best_estimator()
    best_hyperparameters = estimator.hyperparameters()
    logger.info(f'Best hyperparameters found: \n {json.dumps(best_hyperparameters, indent=2)}')
else:
    # Launch a SageMaker Training job by passing s3 path of the training data
    logger.info(f'Launch a SageMaker training job: {training_job_name} using data from {training_dataset_s3_path}')
    estimator.fit(
        inputs={'training': training_dataset_s3_path},
        logs='All',
        job_name=training_job_name,
        wait=True
    )
    logger.info('Training job completed.')


logger.info('Deploy trained model')
# **********************************************************************************************************************
deploy_serverless = True
model_name_ts = name_from_base(model_name, max_length=80)
endpoint_name = model_name_ts
inference_instance_type = "ml.m5.large"

if deploy_serverless:
    serverless_inference_config = ServerlessInferenceConfig()
    initial_instance_count = None
else:
    serverless_inference_config = None
    initial_instance_count = 1

# Retrieve the inference docker container uri
deploy_image_uri = image_uris.retrieve(
    region=None,
    framework=None,
    image_scope="inference",
    model_id=train_model_id,
    model_version=train_model_version,
    instance_type=inference_instance_type,
)
logger.info(f'Inference docker image uri: {deploy_image_uri}')

# Retrieve the inference script uri
deploy_source_uri = script_uris.retrieve(
    model_id=train_model_id,
    model_version=train_model_version,
    script_scope="inference"
)
logger.info(f'Inference script uri: {deploy_source_uri}')

# Use the estimator from the previous step to deploy to a SageMaker endpoint
predictor = estimator.deploy(
    initial_instance_count=initial_instance_count,
    instance_type=inference_instance_type,
    entry_point="inference.py",
    image_uri=deploy_image_uri,
    source_dir=deploy_source_uri,
    endpoint_name=endpoint_name,
    model_name=model_name_ts,
    serverless_inference_config=serverless_inference_config,
)
logger.info(f'Model deployed to endpoint: {endpoint_name}')


# Delete the SageMaker endpoint and the attached resources
#predictor.delete_model()
#predictor.delete_endpoint()

if False:
    s3 = boto3.client('s3')
    s3.download_file('jumpstart-cache-prod-us-east-1',
                     'catboost-training/train-catboost-classification-model.tar.gz',
                     os.path.join(os.getcwd(), 'train-catboost-classification-model.tar.gz'))
