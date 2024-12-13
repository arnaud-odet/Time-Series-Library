import pandas as pd
import numpy as np
import torch
import math

def args_translator(args):
    
    args.ex_file_path = None
    args.val_ex_file_path = None
    args.validate = False
    args.hidden_size = args.d_model
    args.epoch = args.train_epochs
    args.model_save_path = None
    args.checkpoint_path = None
    args.use_checkpoint = False
    args.seed = 0
    args.layer_num = args.e_layers
    args.multi_agent = False
    args.static_agent_drop = False
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.world_size = torch.cuda.device_count()
    
    return args

def usc_data_translator(batch_x, args):
    batch_size = batch_x.shape[0]
    offset = batch_x.shape[-1] - 30
    x = []
    meta_info = []
    lane_data = []
    for i in range(batch_size) :
        
        # Agent data
        agent_data = [batch_x[i, :, p + offset : p + offset +2] for p in range(15)] 
        
        # Meta info
        d = [ ad[-1,:] - ad[-2,:] for ad in agent_data]
        degree = [math.atan2(dp[1], dp[0]) for dp in d]
        prev_positions = torch.cat([torch.Tensor([ad[-1,0], ad[-1,1],ad[-2,0], ad[-2,1]]).reshape(1,4) for ad in agent_data])
        meta_info = torch.cat((torch.Tensor(degree).reshape(15,1), prev_positions),1)
        
        # Lane data
        lane_data = [torch.zeros(19,128)]
        
        # Labels
        labels = torch.ones(15,2)
        
        # Consider
        consider = torch.ones(1)
        
        # Label is valid
        label_is_valid = torch.ones(15)
        
        
        x.append({'agent_data': agent_data, 
            'lane_data': lane_data, 
            'city_name': None, 
            'file_name': None, 
            'origin_labels': None, 
            'labels': labels, 
            'label_is_valid': label_is_valid, 
            'consider': consider, 
            'cent_x': None, 
            'cent_y': None, 
            'angle': None, 
            'meta_info': meta_info})
    
    return x