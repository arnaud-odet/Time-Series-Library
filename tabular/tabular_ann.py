# Dsiplaying only tensorflow error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '2'
import tensorflow as tf
# ann
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.callbacks import EarlyStopping

reg_L1 = L1(l1 = 0.001)
reg_L2 = L2(l2 = 0.001)
reg_L1L2 = L1L2(l1 = 0.3, l2 = 0.3)
dropout_rate = 0.3

nb_unit_grid = [4, 8, 16, 32, 64]
n_hidden_layer_grid = [1,2,3]
batch_size_grid = [8,16,32,64]
l1_grid = [0, 0.1,0.01,0.001]
l2_grid = l1_grid.copy()
dropout_interval = [0,0.5]



def summarize_log_infos(history, X_train, y_train, X_test, y_test, set_up_dict, id):
    
    model = history.model
    
    n_epochs = np.argmin(history.history["val_loss"])
    last_epoch_accuracy_ratio = history.history["val_accuracy"][-1] / history.history["accuracy"][-1]
    last_epoch_loss_ratio = history.history["val_loss"][-1] / history.history["loss"][-1]
    
    train_metrics = model.evaluate(X_train, y_train,verbose = False)
    eval_metrics = model.evaluate(X_test, y_test, verbose = False)

    architecture = []
    for layer in model.layers :
        layer_type = layer.name[:layer.name.find('_')] if '_' in layer.name else layer.name
        if layer_type=='dense':
            architecture.append({"type" : layer_type, 
                                "size":layer.bias.shape[0],
                                "activity_regularizer" : {} if layer.activity_regularizer == None else {key : np.round(float(layer.activity_regularizer.__dict__[key]),4) for key in layer.activity_regularizer.__dict__.keys()},
                                "kernel_regularizer" : {} if layer.kernel_regularizer == None else {key : np.round(float(layer.kernel_regularizer.__dict__[key]),4) for key in layer.kernel_regularizer.__dict__.keys()},
                                "bias_regularizer" : {} if layer.bias_regularizer == None else {key : np.round(float(layer.bias_regularizer.__dict__[key]),4) for key in layer.bias_regularizer.__dict__.keys()},
                                })
        elif layer_type == 'dropout':
            architecture.append({"type" : layer_type, 
                                "droupout_rate": layer.rate})
    exp = {"index" : id, 
           "date" : datetime.date.today(), 
           "eval_loss" : eval_metrics[0], 
           "eval_accuracy" : eval_metrics[1],
           "train_loss" : train_metrics[0], 
           "train_accuracy":train_metrics[1],
           "n_epochs":n_epochs,
           "last_epoch_loss_ratio": last_epoch_loss_ratio,
           "last_epoch_accuracy_ratio": last_epoch_accuracy_ratio,
           "architecture": architecture }
    exp.update(set_up_dict)
    return exp

def log_ann_training(history, X_train, y_train, X_test, y_test, log_file_str, set_up_dict, return_experiment=False):
    if os.path.exists(log_file_str):
        log_df = pd.read_csv(log_file_str, index_col='index')
        id = log_df.index.max()+1
        log_df = log_df.reset_index()
        first = False
    else :
        id = 1
        first = True

    exp = summarize_log_infos(history, X_train, y_train, X_test, y_test, set_up_dict, id)

    exp_df = pd.DataFrame([exp])
    if first :
        log_df = exp_df.set_index('index')
    else :    
        log_df = pd.concat([log_df, exp_df]).set_index('index')
    log_df.to_csv(log_file_str)
    if return_experiment:
        return exp
    
    # ANN

def initialize_random_bin_model(X, n_hidden_layer, nb_unit_grid ,dropout_rate, l1, l2 ,acti_reg, kern_reg, bias_reg):
    # Regularization 
    reg = L1L2(l1 = l1, l2 = l2)
    
    # Architechure
    model = Sequential([Dense(int(random.choice(nb_unit_grid)), input_dim=X.shape[1])])
    if random.choice([True, False]):
        model.add(Dropout(dropout_rate))
    for i in range(n_hidden_layer):
        model.add(Dense(int(random.choice(nb_unit_grid)), activation='relu', activity_regularizer = reg if acti_reg else None, bias_regularizer = reg if bias_reg else None, kernel_regularizer= reg if kern_reg else None))
        model.add(Dropout(dropout_rate))         
    model.add(Dense(1, activation='sigmoid'))
    
    # Optimization
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

def random_search_ann(X_train, y_train, X_val, y_val,log_file_str, nb_unit_grid, n_hidden_layer_grid ,batch_size_grid, l1_grid, l2_grid, dropout_interval, n_iter=1000, verbose=True, return_model:bool=False):

    best_score = 0
    
    for i in range(n_iter) :
        
        batch_size = random.choice(batch_size_grid)
        l1 = float(np.round(l1_grid[np.random.randint(len(l1_grid))] * random.choice(range(10)),4)) 
        l2 = float(np.round(l2_grid[np.random.randint(len(l2_grid))] * random.choice(range(10)),4))
        dropout = np.round(np.random.uniform(dropout_interval[0], dropout_interval[1]),2)
        n_hidden_layer = random.choice(n_hidden_layer_grid)
        
        
        set_up_dict={"batch_size": batch_size}
        model = initialize_random_bin_model(X_train, n_hidden_layer, nb_unit_grid, dropout_rate= dropout, l1= l1, l2= l2, acti_reg=random.choice([True,False]), bias_reg=random.choice([True,False]), kern_reg=random.choice([True,False]))
        es = EarlyStopping(patience=256, restore_best_weights=True, verbose=False)
        history = model.fit(
            X_train,
            y_train,
            validation_split = 0.25,
            #validation_data=[X_val,y_val],
            epochs=16000,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0,
        )
        exp = log_ann_training(history, X_train, y_train, X_val, y_val, log_file_str=log_file_str, set_up_dict= set_up_dict, return_experiment=True)

        if exp['eval_accuracy'] > best_score :
            best_model = model
            best_score = exp['eval_accuracy']    
            best_arch = exp['architecture']

        if verbose :
            display_status_progress(i+1, n_iter, f'Searching ANN - best accuracy so far : {np.round(best_score,3)} - ')
    
    if return_model :
        return best_model, best_arch