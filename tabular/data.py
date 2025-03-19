import numpy as np
import pandas as pd
import os
from tabular.feature_engineering import compute_centroid, compute_polarization, compute_dispersion
from sklearn.preprocessing import StandardScaler

DATA_PATH = './dataset/USC'

def downsample_data(X, downsample_factor:int, verbose :bool = True):
    stride = X.shape[1] // downsample_factor
    if X.shape[1] % downsample_factor != 0 and verbose :
        print(f"Warning : the sequence length is not a multiple of the downsample factor, using the last {int(stride * downsample_factor)} timesteps")
    X_ds = X[:,-stride*downsample_factor:]

    return X_ds.reshape(X.shape[0], downsample_factor, stride).mean(axis = 2)


def split_data(X, group):
    
    assert X.ndim == 3, "Please provide a X of shape (n,t,63)"
    assert X.shape[2] == 63, "Please provide a X of shape (n,t,63)"
    
    x_coord_mask = [False]*3 + [i%4 == 0 and i//4 -1 in group for i in range(X.shape[2]-3)]
    y_coord_mask = [False]*3 + [(i-1)%4 == 0 and (i-1)//4 -1 in group for i in range(X.shape[2]-3)]
    vx_coord_mask = [False]*3 + [(i-2)%4 == 0 and (i-2)//4 -1 in group for i in range(X.shape[2]-3)]
    vy_coord_mask = [False]*3 + [(i-3)%4 == 0 and (i-3)//4 -1 in group for i in range(X.shape[2]-3)]
    
    return X[:,:,x_coord_mask], X[:,:,y_coord_mask], X[:,:,vx_coord_mask], X[:,:,vy_coord_mask]

def load_data(seq_len:int, 
              pred_len:int, 
              off_mask:bool = False,
              downsample_factor:int =4,
              groups:list=None,
              train_split:float=0.7,
              val_split:float=0.15,
              test_split:float=0.15,
              verbose : bool = True):
    
    assert train_split + val_split + test_split == 1 , 'Please specify split summing to 1'
    
    filepath = os.path.join(DATA_PATH,f'PVTO_seq{seq_len}_tar{pred_len}')
    X = np.load(filepath+'_X.npy')
    y = np.load(filepath+'_y.npy')
    
    if off_mask:
        mask = X[:,0,2].astype(bool)
        X = X[mask]
        y = y[mask]

    """
    # Columns order :
    # - 0 : action id
    # - 1 : target
    # - 2 : offense
    # - 3 to 63 : x, y, vx, vy for player 1 to 15
    """
    # Creating y : using last timestamp for coordinate 1 (target)
    y = y[:,-1,1] 
    
    # Computing metrics and downsampling
    stride = X.shape[1] // downsample_factor
    if X.shape[1] % downsample_factor != 0 and verbose:
        print(f"Warning : the sequence length is not a multiple of the downsample factor, using the last {int(stride * downsample_factor)} timesteps") 
    
    if groups is None :
        groups = [list(range(15))]

    data = {}
    for k,group in enumerate(groups) :
        gx,gy,gvx,gvy = split_data(X,group)
        gx_centroid = compute_centroid(gx)
        gx_centroid = downsample_data(gx_centroid,downsample_factor,verbose = False)
        gdisp = compute_dispersion(gx,gy)
        gdisp = downsample_data(gdisp,downsample_factor,verbose = False)
        gpol = compute_polarization(gvx,gvy)    
        gpol = downsample_data(gpol,downsample_factor,verbose = False)
        data[k] = {'players':group, 'x_centroid':gx_centroid, 'dispersion':gdisp, 'polarization':gpol }
    
    # Finding the split indices
    n = X.shape[0]
    train_ind = int(n*train_split) 
    val_ind = int((train_split+val_split)*n)
    
    for k, group in data.items():
        # Computing evolutions for time windows 1, 2, ...,  downsample_factor
        if downsample_factor > 1 :
            for i in range(1,downsample_factor):
                group['x_centroid'][:,i] = group['x_centroid'][:,i] -  group['x_centroid'][:,:i].sum(axis = 1)
                group['dispersion'][:,i] = group['dispersion'][:,i] -  group['dispersion'][:,:i].sum(axis = 1)
                group['polarization'][:,i] = group['polarization'][:,i] -  group['polarization'][:,:i].sum(axis = 1)        
        group['concatenated_data'] = np.concatenate((group['x_centroid'], group['dispersion'], group['polarization']), axis = 1)
    
    # Concatenating and spliting the data
    if len(groups) > 1 :
        concatenated_data = np.concatenate(tuple(group['concatenated_data'] for k, group in data.items()), axis = 1)
    else :
        concatenated_data = data[0]['concatenated_data']

    X_train = concatenated_data[:train_ind]
    y_train = y[:train_ind]
    X_val = concatenated_data[train_ind:val_ind]
    y_val = y[train_ind:val_ind]
    X_test = concatenated_data[val_ind:]
    y_test = y[val_ind:]
    
    # Scaling 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test