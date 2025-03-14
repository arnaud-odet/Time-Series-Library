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

def load_data(seq_len:int, 
              pred_len:int, 
              off_mask:bool = False,
              downsample_factor:int =4,
              train_split:float=0.7,
              val_split:float=0.15,
              test_split:float=0.15):
    
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
    if X.shape[1] % downsample_factor != 0 :
        print(f"Warning : the sequence length is not a multiple of the downsample factor, using the last {int(stride * downsample_factor)} timesteps")    
    x_centroid, _ = compute_centroid(X)
    x_centroid = downsample_data(x_centroid,downsample_factor,verbose = False)
    disp = compute_dispersion(X)
    disp = downsample_data(disp,downsample_factor,verbose = False)
    pol = compute_polarization(X)    
    pol = downsample_data(pol,downsample_factor,verbose = False)
    
    # Finding the split indices
    n = X.shape[0]
    train_ind = int(n*train_split) 
    val_ind = int((train_split+val_split)*n)
    
    # Scaling the features (polarization needs no scaling as it ranges between 0 and 1 by design)
    scaler = StandardScaler()
    scaler.fit(disp[:train_ind])
    scaler.transform(disp)
    x_centroid = x_centroid / 100 # Centroid range in the field lenght, being 100
    
    # Computing evolutions for time windows 1, 2, ...,  downsample_factor
    if downsample_factor > 1 :
        for i in range(1,downsample_factor):
            x_centroid[:,i] = x_centroid[:,i] -  x_centroid[:,:i].sum(axis = 1)
            disp[:,i] = disp[:,i] -  disp[:,:i].sum(axis = 1)
            pol[:,i] = pol[:,i] -  pol[:,:i].sum(axis = 1)
        
    
    # Concatenating and spliting the data
    data = np.concatenate((x_centroid, disp, pol), axis = 1)
    X_train = data[:train_ind]
    y_train = y[:train_ind]
    X_val = data[train_ind:val_ind]
    y_val = y[train_ind:val_ind]
    X_test = data[val_ind:]
    y_test = y[val_ind:]

    return X_train, y_train, X_val, y_val, X_test, y_test