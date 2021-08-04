import logging
import os
from re import L
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict
from torch import nn

import yaml
from tqdm import tqdm

# from utils.logger.general import check_dataset

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def remove_prefix(from_string, prefix=WANDB_ARTIFACT_PREFIX):
    return from_string[len(prefix):]


def get_run_info(run_path):
    run_path = Path(remove_prefix(run_path, WANDB_ARTIFACT_PREFIX))
    run_id = run_path.stem
    project = run_path.parent.stem
    entity = run_path.parent.parent.stem
    model_artifact_name = 'run_' + run_id + '_model'
    return entity, project, run_id, model_artifact_name


class WandbLogger():
    """
    Log training runs, datasets, models, and predictions to Weights & Biases.
    This logger sends information to W&B at wandb.ai. 
    
    By default, this information
    includes hyperparameters, system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    """
    def __init__(self, opt:dict=None, run_id:str=None, job_type="Training"):
        """
        Initialize a Wandb runs or resume a previous run to upload dataset 
        (if `opt.upload_dataset` is True) or monitoring training processes.

        args:
            opt (namespace): Commandline arguments for this run
            run_id (str): Run ID of W&B run to be resumed
            job_type (str): Set job_type for this run
        """
        self.job_type = job_type
        self.wandb, self.wandb_run = wandb, None if not wandb else wandb.run

        if opt is None:
            opt = {
                'project': 'mlops-wandb-demo',
                'entity': None,
                'name': 'exp',
            }
        
        if isinstance(run_id, str) and run_id.startswith(WANDB_ARTIFACT_PREFIX):
            entity, project, run_id, model_artifact_name = get_run_info(run_id)
            model_artifact_name = WANDB_ARTIFACT_PREFIX + model_artifact_name
            
            assert wandb, 'install wandb to resume wandb runs'
            # Resume wandb-artifact:// runs
            self.wandb_run = wandb.init(job_type=job_type,
                                        id=run_id, 
                                        project=opt['project'],
                                        entity=opt['entity'],
                                        resume='allow',)

        elif self.wandb:
            self.wandb_run = wandb.init(config=opt,
                                        resume="allow",
                                        project=opt['project'],
                                        entity=opt['entity'],
                                        name=opt['name'] if opt['name'] != 'exp' else None,
                                        job_type=job_type,
                                        id=run_id,
                                        allow_val_change=True) if not wandb.run else wandb


    def log(self, log_dict: Dict[str, Any], step:int=None):
        if self.wandb_run:
            self.wandb_run.log(data=log_dict, 
                                step=step)


    def watch(self, model:nn.Module, criterion=None, log="gradients", log_freq=1000, idx=None):
        if self.wandb_run:
            self.wandb_run.watch(model, criterion, log, log_freq, idx)

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
    