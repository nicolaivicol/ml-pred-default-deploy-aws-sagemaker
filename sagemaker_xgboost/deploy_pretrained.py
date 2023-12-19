import boto3
import json
import logging
import time
import subprocess
import sagemaker
from sagemaker import image_uris

import config


logger = logging.getLogger('deploy/main.py')
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('s3transfer').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logger.info('START - deploy/main.py')

client = boto3.client(service_name="sagemaker")
runtime = boto3.client(service_name="sagemaker-runtime")
boto_session = boto3.session.Session()
s3 = boto_session.resource('s3')
region = boto_session.region_name
sagemaker_session = sagemaker.Session()
role = config.AWS_SAGEMAKER_ROLE_ARN


logger.info('build tar "model.tar.gz" containing the model file and the custom inference code')
bash_cmd = 'tar -cvpzf ../artifacts/model.tar.gz inference.py -C ../artifacts model.json'
process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()


s3_bucket = sagemaker_session.default_bucket()
logger.info(f'upload "model.tar.gz" to the s3 bucket for model artifacts: {s3_bucket}')
model_artifacts = f's3://{s3_bucket}/model.tar.gz'
res_upload_s3 = s3.meta.client.upload_file('../artifacts/model.tar.gz', s3_bucket, 'model.tar.gz')
logger.debug('response: \n' + json.dumps(res_upload_s3, indent=2))


logger.info(f'get a prebuilt docker image with xgboost from the aws docker registry')
# https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html
image_uri = image_uris.retrieve(
    framework='xgboost',
    region=region,
    version='1.3-1',
    py_version='py3',
    instance_type='ml.c5.large',
)
logger.debug('image uri: ' + image_uri)


# create model
model_name = 'klrn-model-xgb-' + time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())
logger.info(f'create model with name: {model_name}')
res_create_model = client.create_model(
    ModelName=model_name,
    Containers=[
        {
            'Image': image_uri,
            'Mode': 'SingleModel',
            'ModelDataUrl': model_artifacts,
            'Environment': {
                'SAGEMAKER_SUBMIT_DIRECTORY': model_artifacts,
                'SAGEMAKER_PROGRAM': 'inference.py'
            }
        }
    ],
    ExecutionRoleArn=role,
)
logger.debug('response to create_model: \n' + json.dumps(res_create_model, indent=2, default=str))
logger.info('model ARN: ' + res_create_model['ModelArn'])


# create endpoint configuation
epc_name = 'klrn-model-xgb-epc-' + time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())
logger.info(f'create endpoint config with name: {epc_name}')
res_ep_config = client.create_endpoint_config(
    EndpointConfigName=epc_name,
    ProductionVariants=[
        {
            'VariantName': 'xgboostvariant',
            'ModelName': model_name,
            'InstanceType': 'ml.c5.large',
            'InitialInstanceCount': 1
        },
    ],
)
logger.debug('response to create_endpoint_config: \n' + json.dumps(res_ep_config, indent=2, default=str))
logger.info('endpoint configuration ARN: ' + res_ep_config['EndpointConfigArn'])


# create endpoint
endpoint_name = 'klrn-model-xgb-ep-' + time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())
res_create_ep = client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=epc_name,
)
logger.debug('response to create_endpoint: \n' + json.dumps(res_create_ep, indent=2, default=str))
logger.info('endpoint ARN: ' + res_create_ep['EndpointArn'])


logger.info('monitoring status while creating endpoint...')
res_describe_endpoint = client.describe_endpoint(EndpointName=endpoint_name)
while res_describe_endpoint['EndpointStatus'] == 'Creating':
    res_describe_endpoint = client.describe_endpoint(EndpointName=endpoint_name)
    logger.info(res_describe_endpoint['EndpointStatus'])
    time.sleep(30)
logger.debug('response: \n' + json.dumps(res_describe_endpoint, indent=2, default=str))

logger.info('END - deploy/main.py')
