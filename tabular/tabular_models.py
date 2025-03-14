import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import random
import warnings

from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score
from scipy.stats import uniform, loguniform, randint

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Classification models :
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBClassifier

# baselines 
from sklearn.dummy import DummyClassifier, DummyRegressor





def initiate_reg_model_list():
    
    reg_models_names = ["Linear Regression",
            "ElasticNet",
            "SGD ",
            "kNN",
            #"Support Vector - Linear",
            #"Support Vector - Polynomial",
            "Support Vector - RBF",                
            "Decision Tree",
            "Random Forest",
            "ADA Boost ",
            "Gradient Boosting",
            "XGB ",
    ]
    
    reg_models = [LinearRegression(),
            ElasticNet(max_iter=100000),
            SGDRegressor(max_iter=100000),
            KNeighborsRegressor(),
            #SVR(kernel = "linear"),
            #SVR(kernel = "poly"),        
            SVR(kernel = "rbf"),
            DecisionTreeRegressor(),
            RandomForestRegressor(),
            AdaBoostRegressor(),
            GradientBoostingRegressor(),
            XGBRegressor(),
    ]

    reg_hp = [{},
            {'alpha':loguniform(1e-3, 10), 'l1_ratio':uniform(0,1)},
            {'alpha':loguniform(1e-5, 1), 'l1_ratio':uniform(0,1), 'penalty':['l2','l1','elasticnet',None]},
            {'n_neighbors':randint(2,60)},
            #{'kernel':['linear'], 'C':loguniform(1,100)},
            #{'kernel':['poly'], 'C':loguniform(1,100), 'degree':randint(2,3), 'gamma':loguniform(1e-4, 1)},        
            {'kernel':['rbf'], 'C':loguniform(1,100), 'gamma':loguniform(1e-4, 1)},
            {'max_depth': randint(1,20), 'min_samples_leaf' : randint(1,40), 'min_samples_split' : randint(2,20)},
            {'n_estimators':randint(10,500), 'criterion':['squared_error', 'friedman_mse', 'poisson', 'absolute_error'], 'max_depth': randint(1,20), 'min_samples_leaf' : randint(1,40)},
            {'n_estimators':randint(10,500), 'learning_rate': loguniform(1e-3,10)},
            {'n_estimators':randint(10,500), 'learning_rate': loguniform(1e-3,10), 'max_depth': randint(1,10), 'min_samples_leaf' : randint(1,40), 'min_samples_split' : randint(2,20)},
            {'eta':uniform(0,1), 'gamma':loguniform(1e-4,1000), 'max_depth': randint(1,20), 'lambda':loguniform(1e-2,10)},
    ]

    return reg_models_names,reg_models, reg_hp

def initiate_class_model_list():

    clas_models_names = ["Logistic Regression",
                    #"sgd_classifier",
                    "k - Nearest Neighbors Classifier",                
                    #"decision_tree_classifier",
                    "Random Forest",
                    "Ada Boost",
                    #"gradient_boosting_classifier",
                    "XGB Classifier",
                    'Support Vector Classifier - Linear kernel',
                    'Support Vector Classifier - Polynomial kernel',
                    'Support Vector Classifier - RBF kernel',
                    "Partial Least Squares - Discriminant Analysis",                                     
                    "ANN",
                    "Baseline",
    ]

    clas_models = [LogisticRegression(max_iter=1000),
                #SGDClassifier(),
                KNeighborsClassifier(),
                #DecisionTreeClassifier(),
                RandomForestClassifier(),
                AdaBoostClassifier(),
                #GradientBoostingClassifier(),
                XGBClassifier(),
                SVC(kernel = "linear"),
                SVC(kernel = "poly"),        
                SVC(kernel = "rbf"),
                PLSRegression(n_components=11),  
                'initiate_ann_class_simple_model(X_train,n_class)',
                DummyClassifier(),
    ]
    
    clas_hp = [{'penalty':['l1','l2','elasticnet',None], 'C':loguniform(1,1000), 'solver':['saga']},
                #SGDClassifier(),
                {'n_neighbors':randint(2,60)},
                #DecisionTreeClassifier(),
                {'n_estimators':randint(10,500), 'criterion':['gini', 'entropy', 'log_loss'], 'max_depth': randint(1,20), 'min_samples_leaf' : randint(1,40)},
                {'n_estimators':randint(10,500), 'learning_rate': loguniform(1e-3,10)},
                #GradientBoostingClassifier(),
                {'eta':uniform(0,1), 'gamma':loguniform(1e-4,1000), 'max_depth': randint(1,20), 'lambda':loguniform(1e-2,10)},
                {'kernel':['linear'], 'C':loguniform(1,100)},
                {'kernel':['poly'], 'C':loguniform(1,100), 'degree':randint(2,3), 'gamma':loguniform(1e-4, 1)},        
                {'kernel':['rbf'], 'C':loguniform(1,100), 'gamma':loguniform(1e-4, 1)},
                {'n_components':randint(1,8)},  
                {},
                {},
    ]
    
    return clas_models_names,clas_models, clas_hp


# Model comparison

def compare_models_w_hpo(X_train, y_train, X_val, y_val, X_test, y_test, cv:bool = False, n_iter_ml = 100, n_iter_ann = 10,method ='binary', verbose = True):

    different_test_scores = []
    best_HP = []
    asc = method == 'regression'
    kpi = 'RMSE' if method == 'regression' else 'Accuracy'
    
    if method == 'regression' :     
        reg_models_names, reg_models, reg_hp = initiate_reg_model_list()
        for model_name, model, hp in zip(reg_models_names, reg_models, reg_hp):
            if verbose :
                print(f"Testing {model_name} ...", end='\r')
            if model_name[:3] == 'ANN':
                y_pred = np.zeros(y_test.shape)

            else :
                X_train_tmp = np.concatenate((X_train, X_val))
                y_train_tmp = np.concatenate((y_train, y_val))
                split = [-1] * X_train.shape[0] + [0] *X_val.shape[0]
                ps = PredefinedSplit(test_fold=split)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    hrs = RandomizedSearchCV(model, hp, cv = 5 if cv else ps,n_jobs=-1, n_iter = n_iter_ml,refit=True).fit(X_train_tmp, y_train_tmp)
                best_HP.append(hrs.best_params_)
                y_pred = hrs.predict(X_test)

            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            different_test_scores.append(rmse)

            if verbose :
                print(f"Tested {model_name} - reached RMSE of {rmse:.3f} over the test set", end = '\n', flush = True)
            
            comparing_models = pd.DataFrame(list(zip(reg_models_names, best_HP, different_test_scores)),
                                                            columns =['Model', 'Best hyperparameters',kpi])
            
    else : 
        ## Cat
        n_class = len(np.unique(y_train))
        f1s = []
        roc_aucs = []
        cks= []
        clas_models_names, clas_models, clas_hp = initiate_class_model_list(X_train,n_class)
        for model_name, model,hp in zip(clas_models_names, clas_models,clas_hp):
            if verbose :
                print(f"Testing {model_name}", end='\r')
            if model_name[:3] == 'ANN':
                #y_train = to_categorical(y_train)
                """
                es = EarlyStopping(patience=256, restore_best_weights=True, verbose=0)
                model.fit(X_train, 
                    y_train, 
                    validation_data = (X_val, y_val), 
                    epochs = 16000, 
                    batch_size = 32,
                    callbacks = [es],
                    verbose = 0)  
                best_HP.append('na')   
                temp_y_pred = model.predict(X_test)   
                temp_y_pred = np.argmax(temp_y_pred,axis = 1)  
                """
                pass
                """
                model, arch = random_search_ann(
                    X_train = X_train,
                    y_train = y_train,
                    X_val = X_val,
                    y_val = y_val,
                    log_file_str= output_path+'bball_ann_test.csv',
                    nb_unit_grid=nb_unit_grid,
                    n_hidden_layer_grid=n_hidden_layer_grid,
                    batch_size_grid=batch_size_grid,
                    l1_grid=l1_grid,
                    l2_grid=l2_grid,
                    dropout_interval=dropout_interval,
                    n_iter=n_iter_ann,
                    verbose = True,
                    return_model=True
                )  
                # A REVOIR !!!
                best_HP.append(arch)   
                temp_y_pred = model.predict(X_test)   
                #temp_y_pred = np.argmax(temp_y_pred,axis = 1) 
                """     
            else :
                X_train_tmp = np.concatenate((X_train, X_val))
                y_train_tmp = np.concatenate((y_train, y_val))
                split = [-1] * X_train.shape[0] + [0] *X_val.shape[0]
                ps = PredefinedSplit(test_fold=split)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    hrs = RandomizedSearchCV(model, hp, cv = 5 if cv else ps,n_jobs=-1, n_iter = n_iter_ml,refit=True).fit(X_train_tmp, y_train_tmp)
                best_HP.append(hrs.best_params_)
                temp_y_pred = hrs.predict(X_test)
            
            roc_aucs.append(roc_auc_score(y_test, temp_y_pred))
            temp_y_pred = np.array([ 1 if pred > 0.5 else 0 for pred in temp_y_pred])      
            acc = accuracy_score(y_test,temp_y_pred)
            cks.append(cohen_kappa_score(y_test, temp_y_pred))
            different_test_scores.append(acc) 
            f1s.append(f1_score(temp_y_pred,y_test))
            if verbose :
                print(f"Tested {model_name} - reached accuracy of {np.round(acc,4)} over the test set", end = '\n', flush = True)
            comparing_models = pd.DataFrame(list(zip(clas_models_names, best_HP,different_test_scores, f1s, roc_aucs, cks)),
                                                            columns =['Model', 'Best_hyperparameters',kpi, 'F1-score', 'ROC-AUC', 'Cohen-Kappa'])

    return comparing_models.set_index('Model').sort_values(by = kpi,ascending = asc)
