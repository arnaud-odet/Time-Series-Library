from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from utils.log import log
import math
from pathlib import Path

warnings.filterwarnings('ignore')


class Pruning(Exp_Basic):
    def __init__(self, 
                args,
                exp:list,
                pruning_epoch: int=2,
                pruning_factor: float = 0.5,
                ):
        super(Pruning, self).__init__(args)
        try :
            pruning_directory = Path(args.pruning_directory)
        except :
            pruning_directory = Path('./pruning')
        if not os.path.exists(pruning_directory):
            os.mkdir(pruning_directory)
        self.path = pruning_directory
        
        
        
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optimizer == 'adamw':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.wd)         
        else :
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion