# Init logging object
import warnings

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter as TFWriter

from utils.logger.general import colorstr
from utils.logger.wandb.wandb_logger import WandbLogger

try:
    import wandb
    WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'
    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

LOGGER = ('wandb', 'tb')

class SummaryWriter():
    """
    A custome Logger writes event files to be consumed by TensorBoard, Weight & Biases.

    The class updates the contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """
    def __init__(self, log_dir:str=None, config:dict=None):
        """
        Creates a `Looger` that will capture runs metadata, write out events, version artifact
        and summaries to Tensorboard event file and Weight & Biases dashboard.

        Args:
            log_dir (str): Save directory location. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
            config (dict): Dictionary contains hyperparameter
                e.g. config = {'lr': 1e-3, 
                                'weight_decay': 1e-5, 
                                'momentum': 0.7,}

        """
        self.log_dir    = log_dir
        self.config     = config
        self.use_wandb  = wandb is not None
        self.log_prefix = 'Weights & Biases: ' if self.use_wandb else "Tensorboard: "
        # Message
        if not self.use_wandb:
            self.log_message("run 'pip install wandb' to automatically track and visualize runs.")
    
        if self.use_wandb:
            self.__init_wandb()
        else:
            self.__init_tensorboard()

    def log_message(self, message:str, prefix:str=None):
        if prefix is None:
            prefix = self.log_prefix

        prefix = colorstr(prefix)
        s = f"{prefix}{message}"
        print(str(s))
    
    def __init_tensorboard(self,):
        self.log_message(f"Start with 'tensorboard --logdir {self.log_dir}', view at http://localhost:6006/")
        self.tensorboard = TFWriter(str(self.log_dir))

    def __init_wandb(self,):
        # wandb_artifact_resume = isinstance(self.opt.weight, str) and self.opt.weight.startswith(WANDB_ARTIFACT_PREFIX)
        # run_id = torch.load(self.opt.weights) if not wandb_artifact_resume else None
        self.wandb = WandbLogger()

    def get_logdir(self):
        """Return directory"""
        return self.log_dir

    def watch_model(self, model:nn.Module, criterion=None, log="gradients", log_freq=1000, idx=None):
        if self.use_wandb:
            self.wandb.watch(model, criterion, log, log_freq, idx)
        else:
            self.log_message("Does not support watch model with Tensorboard, please use wandb")
    
    def add_scalar(self, tag:str, scalar_value, global_step=None):
        """
        log(
            data: Dict[str, Any],
            step: int = None,
            commit: bool = None,
            sync: bool = None
        ) -> None
        """
        if self.use_wandb:
            self.wandb.log({tag:scalar_value}, step=global_step)
        else:
            self.tensorboard.add_scalar(tag, scalar_value, global_step)

    def add_scalars(self, main_tag, tag_scalar_dict:dict, global_step=None, walltime=None):
        """
        Usage:
            main_tag = 'train'
            for i in range(10):
                tag_scalar_dict = {
                    'loss_cls' = i/10,
                    'loss_bbox' = i/10, 
                }
                add_scalars(main_tag, tag_scalar_dict, global_step=i)
                
        """
        if self.use_wandb:
            wb_scalar_dict = {}
            for key, value in tag_scalar_dict.items():
                wb_scalar_dict[str(main_tag+'/'+key)] = value
                
            self.wandb.log(wb_scalar_dict, step=global_step)
        else:
            self.tensorboard.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def log_dataset_artifact():
        pass

    def download_dataset_artifact():
        pass

    def log_model_artifact():   
        pass

    def download_model_artifact():
        pass