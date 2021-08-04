# Init logging object
from utils.logger.wandb.wandb_logger import WandbLogger
from utils.logger.general import colorstr
import warnings

import torch
from torch.utils.tensorboard import SummaryWrite

try:
    import wandb
    WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'
    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

LOGGER = ('wandb', 'tb')

class Hparams():
    """
    """
    def __init__(self):
        pass

class Logger():
    """
    A custome Logger writes event files to be consumed by TensorBoard, Weight & Biases.

    The class updates the contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """
    def __init__(self, log_dir=None, opt=None, hyp=None, logger=None, include=LOGGER):
        """
        Creates a `Looger` that will capture runs metadata, write out events, version artifact
        and summaries to Tensorboard event file and Weight & Biases dashboard.

        Args:
            log_dir (str): Save directory location. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
            hyp (dict): A Runs hyperparameters.
                e.g: {'lr': 1e-5,
                        'weight_decay': 0.7,
                        'momentum': 0.3,}
            opt (dict): Runs arguments contain 
                - epochs (int): number of epochs
                - batch_size (int): size of the batch
                - weights (str): name of pretrain/resume 
                - data (dict): dataset information 
                - resume (boolean):
            logger (function): Logger for printing results to console.

        """
        self.log_dir    = log_dir
        self.hyp        = hyp
        self.logger     = logger
        self.include    = include
        self.keys       = [ 'train/loss', # train loss
                            'metrics/prcision', 'metrics/recall', 'metrics/auc', # metrics
                            'val/loss', # val loss
                            'x/lr', ] # params
        self.csv        = True # always save results to csv
        for k in self.include:
            setattr(self, k, None) # init empty logger dictionary

        # Message
        if not wandb:
            prefix = colorstr('Weigts & Biases: ')
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize runs."
            print(str(s))
            
        # Tensorboard
        s = self.log_dir
        if 'tb' in self.include:
            prefix = colorstr('Tensorboard: ')
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWrite(str(s))

        if wandb and 'wandb' in self.include:
            wandb_artifact_resume = isinstance(self.opt.weight, str) and self.opt.weight.startswith(WANDB_ARTIFACT_PREFIX)
            run_id = torch.load(self.weights.get('wandb_id')) if not wandb_artifact_resume else None
            self.wandb = WandbLogger(self.opt, run_id)

        else:
            self.wandb=None
        

    def add_hparams(self):
        pass

    def add_scalar(self):
        pass

    def add_scalars(self):
        pass
