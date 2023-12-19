import os
import boto3
import json

AWS_SAGEMAKER_ENDPOINT_MODEL = os.environ['AWS_SAGEMAKER_ENDPOINT_MODEL']

runtime = boto3.client('runtime.sagemaker')


def lambda_handler(event, context):
    data = json.loads(json.dumps(event))
    payload = data['data']
    response = runtime.invoke_endpoint(
        EndpointName=AWS_SAGEMAKER_ENDPOINT_MODEL,
        ContentType='text/csv',
        Body=payload)
    result = json.loads(response['Body'].read().decode())
    return result
