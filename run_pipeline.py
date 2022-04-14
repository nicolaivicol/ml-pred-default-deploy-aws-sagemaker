from select_feats import select_feats
from tune_hyper_params import tune_hyper_params
from cv import cv
from train import train
from predict import predict
import logging


def run_pipeline():
    """
    Run the full pipeline which includes: \n
    - feature selection
    - hyper-parameters tuning via CV
    - cross-validation with best parameters to estimate model's performance
    - train model with best parameters
    - predict on data provided locally
    """
    logger = logging.getLogger('run_pipeline.py')
    logger.info('START - Run pipeline')
    select_feats()
    tune_hyper_params()
    cv()
    train()
    predict()
    logger.info('END - Run pipeline')


if __name__ == "__main__":
    run_pipeline()
