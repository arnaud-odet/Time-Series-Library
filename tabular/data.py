import numpy as np
import pandas as pd
import os

DATA_PATH = './dataset/USC'

def load_data(seq_len:int, 
              pred_len:int, 
              off_mask:bool = False,
              train_split:float=0.7,
              val_split:float=0.15,
              test_split:float=0.15):
    
    filepath = os.path.join(DATA_PATH,f'PVTO_seq{seq_len}_tar{pred_len}')
    X = np.load(filepath+'_X.npy')
    y = np.load(filepath+'_y.npy')
    
    if off_mask:
        mask = X[:,0,2].astype(bool)
        X = X[mask]
        y = y[mask]

    # Columns order :
    # 0 : action id
    # 1 : target
    # 2 : offense
    # 3 to 63 : x, y, vx, vy for player 1 to 15
    
    y = y[:,-1,1] # Last timestamp for coordinate 1 (target)

    return X, y