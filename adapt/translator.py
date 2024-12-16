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
    args.multi_agent = True
    args.static_agent_drop = False
    args.first_pass_verbose = True
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.world_size = torch.cuda.device_count()
    
    return args

def usc_data_translator(batch_x, args, verbose :bool = True):
    batch_size = batch_x.shape[0]
    offset = batch_x.shape[-1] - 30
    x = []
    meta_info = []
    lane_data = []
    for i in range(batch_size) :
        
        # Agent data
        pad = torch.zeros((batch_x.shape[1], args.hidden_size - 2)).to(args.device)
        agent_data = [torch.cat((batch_x[i, :, p + offset : p + offset +2], pad),1)  for p in range(15)] 
        if verbose and args.first_pass_verbose:
            print("=== For each scene in batch ===")
            print(f"Agent data : len = {len(agent_data)} of shape {agent_data[0].shape}")

        # Lane data
        lane_data = [torch.zeros(1,args.hidden_size)]
        if verbose and args.first_pass_verbose:
            print(f"Lane data : len = {len(lane_data)} of shape {lane_data[0].shape}")
        
        # Meta info
        d = [ ad[-1,:2] - ad[-2,:2] for ad in agent_data]
        degree = [math.atan2(dp[1], dp[0]) for dp in d]
        prev_positions = torch.cat([torch.Tensor([ad[-1,0], ad[-1,1],ad[-2,0], ad[-2,1]]).reshape(1,4) for ad in agent_data])
        meta_info = torch.cat((torch.Tensor(degree).reshape(15,1), prev_positions),1)
        if verbose and args.first_pass_verbose:
            print(f"Meta info : shape {meta_info.shape}")
                
        # Labels
        labels = torch.ones(args.pred_len,15,2)
        if verbose and args.first_pass_verbose:
            print(f"Labels : shape {labels.shape}")
        
        # Consider
        consider = torch.ones(args.pred_len).long()
        if verbose and args.first_pass_verbose:
            print(f"Consider : shape {consider.shape}")
        
        # Label is valid
        label_is_valid = torch.ones(args.pred_len,15)
        if verbose and args.first_pass_verbose:
            print(f"Label is valid : shape {label_is_valid.shape}")
            args.first_pass_verbose = False

        
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