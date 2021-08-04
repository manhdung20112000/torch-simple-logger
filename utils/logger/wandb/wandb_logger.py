import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import yaml
from tqdm import tqdm

from utils.logger.general import check_dataset

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

class WandbLogger():
    """
    Log training runs, datasets, models, and predictions to Weights & Biases.
    This logger sends information to W&B at wandb.ai. 
    
    By default, this information
    includes hyperparameters, system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    """
    def __init__(self, opt, run_id, job_type="Training"):
        """
        Initialize a Wandb runs or resume a previous run to upload dataset 
        (if `opt.upload_dataset` is True) or monitoring training processes.

        args:
            opt (namespace): Commandline arguments for this run
            run_id (str): Run ID of W&B run to be resumed
            job_type (str): Set job_type for this run
        """
        pass


    def setup_training(self, opt):
        """

        """
        pass


    def download_dataset_artifact(self, path, alias):
        """
        """
        pass


    def download_model_artifact(self, opt):
        """
        """
        pass


    def log_dataset_artifact(self, data_file, project, overwrite_config=False):
        """
        """
        pass


    def log_model(self, path, opt, epoch, best_model=False):
        """
        """
        pass

    def log_training_process(self):
        """
        """
        pass
    