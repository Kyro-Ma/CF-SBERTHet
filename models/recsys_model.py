import contextlib
import logging
import os
import glob
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from utils import *
from pre_train.sasrec.model import SASRec


def load_checkpoint(recsys, pre_trained):
    path = f'pre_train/{recsys}/{pre_trained}/'
    '''
    A .pth file in PyTorch is a checkpoint file that stores either:
    Model weights (i.e., state_dict)
    Full model (architecture + weights) – less common (and not recommended)
    Custom objects like dictionaries with multiple parts:
        model weights (state_dict)
        optimizer state
        training args or configs (like your kwargs and args)
        epoch, loss history, etc.
    '''
    pth_file_path = find_filepath(path, '.pth')
    assert len(pth_file_path) == 1, 'There are more than two models in this dir. You need to remove other model files.\n'
    kwargs, checkpoint = torch.load(pth_file_path[0], map_location="cpu", weights_only=False)
    logging.info("load checkpoint from %s" % pth_file_path[0])

    return kwargs, checkpoint


class RecSys(nn.Module):
    def __init__(self, recsys_model, pre_trained_data, device='cuda'):
        super().__init__()
        '''
        kwargs (short for keyword arguments) is a dictionary that contains the model's architecture 
        configuration and hyperparameters, such as the number of hidden units, item/user count, 
        dropout rates, number of layers, and other settings required to reconstruct the model 
        structure. These are the parameters that would typically be passed into the model's 
        __init__() method when instantiating it (e.g., model = SASRec(**kwargs)).

        checkpoint is a state_dict, which is a dictionary storing the actual weights and biases 
        (tensors) of the model that were learned during training. This dictionary maps layer names 
        to tensors and is used to load the trained state into the model (e.g., 
        model.load_state_dict(checkpoint)).
        '''
        kwargs, checkpoint = load_checkpoint(recsys_model, pre_trained_data)
        kwargs['args'].device = device
        model = SASRec(**kwargs)
        model.load_state_dict(checkpoint)
            
        for p in model.parameters():
            p.requires_grad = False
            
        self.item_num = model.item_num
        self.user_num = model.user_num
        self.model = model.to(device)
        self.hidden_units = kwargs['args'].hidden_units
        
    def forward(self):
        print('forward')