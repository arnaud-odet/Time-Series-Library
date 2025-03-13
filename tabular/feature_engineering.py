import pandas as pd
import numpy as np

def centroid(X):
    
    assert X.ndim == 3, "Please provide a X of shape (n,t,63)"
    assert X.shape[2] == 63, "Please provide a X of shape (n,t,63)"

    x_coord_mask = [False]*3 + [i%4 == 0 for i in range(X.shape[2]-3)]
    y_coord_mask = [False]*3 + [(i-1)%4 == 0 for i in range(X.shape[2]-3)]

    X_x_centoid = X[:,:,x_coord_mask].mean(axis = 2)
    X_y_centoid = X[:,:,y_coord_mask].mean(axis = 2)
    
    return X_x_centoid, X_y_centoid  

def dispersion(X):

    assert X.ndim == 3, "Please provide a X of shape (n,t,63)"
    assert X.shape[2] == 63, "Please provide a X of shape (n,t,63)"

    x_coord_mask = [False]*3 + [i%4 == 0 for i in range(X.shape[2]-3)]
    y_coord_mask = [False]*3 + [(i-1)%4 == 0 for i in range(X.shape[2]-3)]

    X_x_centoid, X_y_centoid = centroid(X)

    disp = np.sqrt(
        ((X[:,:,x_coord_mask] - X_x_centoid[:,:,np.newaxis].repeat(15, axis = 2))**2 +
        (X[:,:,y_coord_mask] - X_y_centoid[:,:,np.newaxis].repeat(15, axis = 2))**2).mean(axis = 2)
    )
    
    return disp