import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from data_provider.data_factory import data_provider
from types import SimpleNamespace
from sklearn.metrics import mean_squared_error

DATA_PATH = "./dataset/USC"
MODEL_PATH = Path('./pruning/top_exps')
PERMUTATIONS_PATH = Path('./permutations')
BEST_EXPS = {16: '7_055', 
            #  32: '1_050', 
             32: '1_153',
             64: '27_034', 
             128: '21_113'}
SELECTED_EXPS_1 = {16: '7_033', 
             32: '9_033',
             64: '11_163', 
             128: '13_099'}
SELECTED_EXPS_2 = {16: '15_151', 
             32: '9_018',
             64: '11_025', 
             128: '21_083'}
SELECTED_EXPS_3 = {16: '7_070', 
             32: '25_078',
             64: '3_149', 
             128: '21_113'}
SELECTED_EXPS = {1: SELECTED_EXPS_1, 2:SELECTED_EXPS_2, 3: SELECTED_EXPS_3}



def create_dataloader(th:int, flag:str='test'):
    dp_args = SimpleNamespace(**{"root_path" : DATA_PATH, 
                                        'data_path' : 'na',
                                        "data":'USC', 
                                        'seq_len':th, 
                                        'pred_len':th, 
                                        'label_len':0,
                                        'scale':True,
                                        # Default args to allow loader to work
                                        'model':'Transformer',
                                        'task_name':'long_term_forecast',
                                        'embed':'timeF',
                                        'features':'MS',
                                        'use_action_progress':True,
                                        'use_offense':False,
                                        'consider_only_offense':True,
                                        'batch_size':128,
                                        'freq':'h',
                                        'seasonal_patterns':'Monthly',
                                        'num_workers' : 2,
                                        'target':'na'})
    _, uscdl = data_provider(args=dp_args, flag = flag)
    return uscdl

def load_model(th:int, exps):
    exp_path = MODEL_PATH / exps[th] / 'best.pt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(exp_path, map_location=torch.device(device))
    ### Dynamically import and instantiate the model class
    module_name = checkpoint['model_module']
    class_name = checkpoint['model_class']
    model_args = checkpoint['model_args']
    ### Import the module and get the class
    module = __import__(module_name, fromlist=[class_name])
    model_class = getattr(module, class_name)
    ### Create model instance using saved arguments
    model = model_class(model_args)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model

def evaluate_model(model, data_loader, metric_fn):
    """Evaluate model on data_loader"""
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, targets, _, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs, None, None, None)
            all_preds.append(outputs.cpu())
            all_targets.append(targets)
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    return metric_fn(targets, preds)


def evaluate_with_permuted_feature(model, data_loader, metric_fn, 
                                   feature_idx):
    """Evaluate with one feature permuted"""
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, targets, _, _ in data_loader:
            # Permute the feature across the batch
            permuted_inputs = inputs.clone()
            perm_idx = torch.randperm(inputs.size(0))
            permuted_inputs[:, :, feature_idx] = inputs[perm_idx, :, feature_idx]
            
            permuted_inputs = permuted_inputs.to(device)
            outputs = model(permuted_inputs, None, None ,None)
            all_preds.append(outputs.cpu())
            all_targets.append(targets)
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    return metric_fn(targets, preds)

def permutation_importance(model, data_loader, metric_fn, n_repeats=10):
    """
    Compute permutation importance for each feature
    Args:
        model: PyTorch model
        data_loader: DataLoader with validation data
        metric_fn: function(y_true, y_pred) -> score
        n_repeats: number of permutation repeats
    Returns:
        importance: shape (features,)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get baseline score
    baseline_score = evaluate_model(model, data_loader, metric_fn)
    
    # Get data shape
    sample_batch = next(iter(data_loader))
    _, _, n_features = sample_batch[0].shape
    
    importance_scores = np.zeros(n_features)
    
    for feature_idx in range(n_features):
        scores = []
        print(f"    Processing feature n° {feature_idx+1:>3} out of {n_features:>3} ...", end = '\r')
        for _ in range(n_repeats):
            # Permute feature across all samples
            permuted_score = evaluate_with_permuted_feature(
                model, data_loader, metric_fn, feature_idx
            )
            scores.append(baseline_score - permuted_score)
        mean_score = np.mean(scores)
        importance_scores[feature_idx] = mean_score
        print(f"    Processing feature n° {feature_idx+1:>3} out of {n_features:>3} | importance score: {mean_score:.2e} ")
        
    
    return importance_scores

def final_displacement_error(true, preds):
    return mean_squared_error(true[:,-1], preds[:,-1])

def permutation_pipeline(th:int, n_repeats:int= 10, version:int=0, which:str='best', flag:str='test'):
    exps = BEST_EXPS if which == 'best' else SELECTED_EXPS[int(which[-1])]
    print(f"    Permutation version n°{version} - {n_repeats} iterations")
    print(f"    Loading model ...", end = '\r')
    model = load_model(th=th, exps=exps)
    print(f"    Loading model COMPLETED")
    print(f"    Creating DataLoader ...", end = '\r')
    dataloader = create_dataloader(th=th, flag= flag)
    print(f"    Creating DataLoader COMPLETED")
    perm_imp = permutation_importance(model = model, data_loader=dataloader, metric_fn=final_displacement_error, n_repeats= n_repeats)
    np.save(PERMUTATIONS_PATH / f"{exps[th]}_v{version}_{n_repeats}.npy", perm_imp)
    return perm_imp

if __name__ == '__main__':
    for th in [16,32,64,128]:
        print(f"Processing time horizon {th:>3}-{th:>3}.")
        permutation_pipeline(th, n_repeats=5, version=12, which = 'selected_v3', flag = 'test')
        