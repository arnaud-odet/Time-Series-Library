import pandas as pd
import numpy as np

def compute_centroid(x):
    
    return x.mean(axis = 2) 

def compute_dispersion(x,y):

    x_centroid = compute_centroid(x)
    y_centroid = compute_centroid(y)

    disp = np.sqrt(
        ((x - x_centroid[:,:,np.newaxis].repeat(x.shape[2], axis = 2))**2 +
        (y - y_centroid[:,:,np.newaxis].repeat(y.shape[2], axis = 2))**2).mean(axis = 2)
    )
    
    return disp

def compute_polarization(vx,vy):

    velocity_norm = np.sqrt((vx**2 + vy**2))
    uvx = vx / velocity_norm
    uvy = vy / velocity_norm
    uvx = np.nan_to_num(uvx, copy=False)
    uvy = np.nan_to_num(uvy, copy=False)

    polarization = np.sqrt(uvx.mean(axis = 2)**2 + uvy.mean(axis = 2)**2)
    return polarization