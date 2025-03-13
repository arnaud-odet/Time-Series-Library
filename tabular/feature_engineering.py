import pandas as pd
import numpy as np

def compute_centroid(X):
    
    assert X.ndim == 3, "Please provide a X of shape (n,t,63)"
    assert X.shape[2] == 63, "Please provide a X of shape (n,t,63)"

    x_coord_mask = [False]*3 + [i%4 == 0 for i in range(X.shape[2]-3)]
    y_coord_mask = [False]*3 + [(i-1)%4 == 0 for i in range(X.shape[2]-3)]

    X_x_centoid = X[:,:,x_coord_mask].mean(axis = 2)
    X_y_centoid = X[:,:,y_coord_mask].mean(axis = 2)
    
    return X_x_centoid, X_y_centoid  

def compute_dispersion(X):

    assert X.ndim == 3, "Please provide a X of shape (n,t,63)"
    assert X.shape[2] == 63, "Please provide a X of shape (n,t,63)"

    x_coord_mask = [False]*3 + [i%4 == 0 for i in range(X.shape[2]-3)]
    y_coord_mask = [False]*3 + [(i-1)%4 == 0 for i in range(X.shape[2]-3)]

    X_x_centoid, X_y_centoid = compute_centroid(X)

    disp = np.sqrt(
        ((X[:,:,x_coord_mask] - X_x_centoid[:,:,np.newaxis].repeat(15, axis = 2))**2 +
        (X[:,:,y_coord_mask] - X_y_centoid[:,:,np.newaxis].repeat(15, axis = 2))**2).mean(axis = 2)
    )
    
    return disp

def compute_polarization(X):

    assert X.ndim == 3, "Please provide a X of shape (n,t,63)"
    assert X.shape[2] == 63, "Please provide a X of shape (n,t,63)"

    vx_coord_mask = [False]*3 + [(i-2)%4 == 0 for i in range(X.shape[2]-3)]
    vy_coord_mask = [False]*3 + [(i-3)%4 == 0 for i in range(X.shape[2]-3)]

    velocity_norm = np.sqrt((X[:,:,vx_coord_mask]**2 + X[:,:,vy_coord_mask]**2))
    uvx = X[:,:,vx_coord_mask] / velocity_norm
    uvy = X[:,:,vy_coord_mask] / velocity_norm
    uvx = np.nan_to_num(uvx, copy=False)
    uvy = np.nan_to_num(uvy, copy=False)

    polarization = np.sqrt(uvx.mean(axis = 2)**2 + uvy.mean(axis = 2)**2)
    return polarization