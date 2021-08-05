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

from utils.logger.general import check_dataset

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

        if self.wandb_run:
            if self.job_type == 'Training':
                if not opt.resume:
                    if opt.upload_dataset:
                        self.wandb_artifact_data_dict = self.check_and_upload_dataset(opt)

                    elif opt.data.endswith('_wandb.yaml'):
                        with open(opt.data, encoding='ascii', errors='ignore') as f:
                            data_dict = yaml.safe_load(f)
                        self.data_dict = data_dict
                    else: # Local .yaml datset file or .zip file
                        self.data_dict = check_dataset(opt.data)
                    
                self.setup_training(opt)
                if not self.wandb_artifact_data_dict:
                    self.wandb_artifact_data_dict = self.data_dict

                if not opt.weights.startswith(WANDB_ARTIFACT_PREFIX):
                    self.wandb_run.config.update({'data_dict': self.wandb_artifact_data_dict})
                    pass


    def log(self, log_dict: Dict[str, Any], step:int=None):
        if self.wandb_run:
            self.wandb_run.log(data=log_dict, 
                                step=step)


    def watch(self, model:nn.Module, criterion=None, log="gradients", log_freq=1000, idx=None):
        if self.wandb_run:
            self.wandb_run.watch(model, criterion, log, log_freq, idx)

    
    def check_and_upload_dataset(self, opt):
        """
        Check if the dataset format is compatible and upload it as W&B artifact

        Args:
            opt (namespace): Commandline arguments for current run

        Returns:
            Updated dataset info dictionary where local dataset paths are replaced 
              by WAND_ARFACT_PREFIX links.
        """
        assert wandb, 'Install wandb to upload dataset'
        config_path = self.log_dataset_artifact(opt.data, opt.project)
        print("Created dataset config file ", config_path)
        with open(config_path, encoding='ascii', errors='ignore') as f:
            wandb_data_dict = yaml.safe_load(f)
        return wandb_data_dict

    def setup_training(self, opt):
        """
        Setup the necessary processes for training models:
        - Attempt to download model checkpoint and dataset artifact (if opt.weights 
          or opt.dataset starts with WANDB_ARTIFACT_PREFIX)
        - Update data_dict, to contain info of previous run if resumed and the paths 
          of dataset artifact if downloaded
        """
        self.log_dict = {}
        if isinstance(opt.weights, str) and opt.weights.startswith(WANDB_ARTIFACT_PREFIX):
            model_dir = _ = self.download_model_artifact(opt)
            if model_dir:
                self.weights = Path(model_dir)
                config = self.wandb_run.config 
                
                # TODO: define opt format
            
        else:
            data_dict = self.data_dict
        if self.val_artifact is None: # If --upload_dataset is set, use the existing artifact
            self.train_artifact_path, self.train_artifact = self.download_dataset_artifact(data_dict.get('train'),
                                                                                           opt.artifact_alias)
            self.val_artifact_path, self.val_artifact = self.download_dataset_artifact(data_dict.get('val'),
                                                                                       opt.artifact_alias)

        if self.train_artifact_path is not None:
            train_path = Path(self.train_artifact_path)
            data_dict['train'] = str(train_path)

        if self.val_artifact_path is not None:
            val_path = Path(self.val_artifact_path)  
            data_dict['val'] = str(val_path)

        # TODO: init object for summary at the end of training process
        
        train_from_artifact = self.train_artifact_path is not None and self.val_artifact_path is not None
        if train_from_artifact:
            self.data_dict = data_dict

    def download_dataset_artifact(self, path, alias):
        """
        Download the dataset artifact if the path starts with WANDB_ARTIFACT_PREFIX

        Args:
            path (Path): path of the dataset to be used for training
            alias (str): alias of the artifact to be download/used for training

        Returns:
            (str, wandb.Artifact): path of the downladed dataset and it's corresponding 
              artifact object if dataset is found otherwise returns (None, None)
        
        """
        if isinstance(path, str) and path.startswith(WANDB_ARTIFACT_PREFIX):
            artifact_path = Path(remove_prefix(path, WANDB_ARTIFACT_PREFIX) + f":{alias}")
            dataset_artifact = wandb.use_artifact(artifact_path.as_posix().replace("\\", "/"))
            assert dataset_artifact is not None, "'Error: W&B dataset artifact doesn\'t exist'"
            data_dir = dataset_artifact.download()
            return data_dir, dataset_artifact
        return None, None


    def download_model_artifact(self, opt):
        """
        Download the model checkpoint artifact if the weigth 
        start with WANDB_ARTIFACT_PREFIX

        Args:
            opt (namespace): Comandline arguments for this run
        """
        if isinstance(opt.weights, str) and opt.weights.startswith(WANDB_ARTIFACT_PREFIX):
            model_artifact = wandb.use_artifact(remove_prefix(opt.ưeights, WANDB_ARTIFACT_PREFIX)+":latest")
            assert model_artifact is not None, 'Error: W&B model artifact doesn\'t exist'
            model_dir = model_artifact.download()
            return model_dir, model_artifact
        
        return None, None


    def log_dataset_artifact(self, data_file, project, overwrite_config=False):
        """
        Log the dataset as W&B artifact and return the new data file with W&B links

        Args:
            data_file (str): 
            project (str): 
            overwrite_config (boolean): 

        Returns:
            the new .yaml file with artifact link (which can be used to training directly)
        """
        self.data_dict = check_dataset(data_file)
        
        data = dict(self.data_dict)
        self.train_artifact = self.create_dataset_artifact(data['train'], name='train') if data.get('train') else None
        self.val_artifact = self.create_dataset_artifact(data['val'], name='val') if data.get('val') else None

        if data.get('train'):
            data['train'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'train')
        if data.get('val'):
            data['val'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'val')

        path = Path(data_file).stem
        path = (path if overwrite_config else path + '_wandb') + '.yaml'  # updated data.yaml path
        data.pop('download', None)
        data.pop('path', None)
        with open(path, 'w') as f:
            yaml.safe_dump(data, f)

        if self.job_type == 'Training': 
            self.wandb_run.use_artifact(self.val_artifact)
            self.wandb_run.use_artifact(self.train_artifact)
            # self.val_artifact.wait()
            # self.val_table = self.val_artifact.get('val')
            # self.map_val_table_path()
        else:
            self.wandb_run.log_artifact(self.train_artifact)
            self.wandb_run.log_artifact(self.val_artifact)
        return path


    def create_dataset_artifact(self, path, name='dataset', type='dataset'):
        """
        Create and return W&B artifact containing W&B Table of the dataset.
        
        Args:
            path (Path): Path to dataset dir/file
            name (str) -- name of the artifact
        
        Returns: 
            Dataset artifact to be logged or used
        """
        artifact = wandb.Artifact(name=name, type=type) 
        if isinstance(path, str) and os.path.exists(path):
            if os.path.isdir(path):
                artifact.add_dir(path)
            elif os.path.isfile(path):
                artifact.add_file(path)
        
        return artifact


    def log_model(self, path, opt, epoch, score):
        """
        Log the model checkpoint as W&B artifact

        Args:
            path (Path): Path to the checkpoints file
            opt (namespace): Comand line arguments for this run
            epoch (int): Current epoch number
            score (float): score for current epoch
        """
        model_artifact = wandb.Artifact('run_' + self.wandb_run.id + '_model', type='model',
                                metadata={'project':opt.project,
                                'epochs_trained': epoch+1,
                                'total_epochs': opt.epochs,
                                'score': score})
        model_artifact.add_file(str(path))
        # logging
        self.wandb_run.log_artifact(model_artifact,
                                    aliases=['latest', 'epoch '+ str(epoch+1)])
        print(f"Saving model on epoch {epoch} done.")


    def log_training_process(self):
        """
        """
        pass
    