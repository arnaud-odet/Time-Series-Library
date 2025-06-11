from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate_pruning, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

import copy
from pathlib import Path
import pandas as pd
import json
import csv
import math

warnings.filterwarnings('ignore')


class Pruning(Exp_Basic):
    
    def __init__(self, 
                args
                ):
        """
        Run pruning over the provided exps.
        Exp must be passed as a list of dict where each dict keys contains model hyperparameters of the corresponding experiment.
        """
        super(Pruning, self).__init__(args)
        self.pr_id = args.pruning_id
        try :
            pruning_directory = Path(args.pruning_directory)
        except :
            pruning_directory = Path('./pruning')
        if not os.path.exists(pruning_directory):
            os.mkdir(pruning_directory)
            os.mkdir(pruning_directory / 'learning_curves')
        self.path = pruning_directory
        self.train_logs_file = pruning_directory / f"train_logs_{self.pr_id}.csv"
        self.pruning_epochs = args.pruning_epochs
        self.pruning_factor = args.pruning_factor
        self.model_registry = {}
        self.pruning_config_file = args.pruning_config_file
        if args.is_training :
            with open(self.pruning_config_file) as f:
                experiments = json.load(f)
            self.exps = self._build_exps(experiments)
            self.n_exp_alive = len(self.exps)
            self.print_args() 
            self._resume()
        
    def print_args(self):
        print("\033[1m" + "Basic Config" + "\033[0m")
        print(f'  {"Task Name:":<20}{self.args.task_name:<20}{"Is Training:":<20}{self.args.is_training:<20}')
        print(f'  {"Pruning epochs:":<20}{self.pruning_epochs:<20}{"Pruning factor:":<20}{self.pruning_factor:<20}')
        print(f'  {"N experiments:":<20}{self.n_exp_alive:<20}{"Config_list:":<20}{self.pruning_config_file:<20}')
        print()

        print("\033[1m" + "Data" + "\033[0m")
        print(f'  {"Data:":<20}{self.args.data:<20}{"Features:":<20}{self.args.features:<20}')
        print(f'  {"Seq Len:":<20}{self.args.seq_len:<20}{"Label Len:":<20}{self.args.label_len:<20}')
        print(f'  {"Pred Len:":<20}{self.args.pred_len:<20}{"Inverse:":<20}{self.args.inverse:<20}')
        print()

        print("\033[1m" + "Run Parameters" + "\033[0m")
        print(f'  {"Num Workers:":<20}{self.args.num_workers:<20}{"Itr:":<20}{self.args.itr:<20}')
        print(f'  {"Train Epochs:":<20}{self.args.train_epochs:<20}{"Batch Size:":<20}{self.args.batch_size:<20}')
        print(f'  {"Patience:":<20}{self.args.patience:<20}{"Learning Rate:":<20}{self.args.learning_rate:<20}')
        print(f'  {"Use Amp:":<20}{self.args.use_amp:<20}{"Loss:":<20}{self.args.loss:<20}')
        print()

        print("\033[1m" + "GPU" + "\033[0m")
        print(f'  {"Use GPU:":<20}{self.args.use_gpu:<20}{"GPU:":<20}{self.args.gpu:<20}')
        print(f'  {"Use Multi GPU:":<20}{self.args.use_multi_gpu:<20}{"Devices:":<20}{self.args.devices:<20}')
        print()      
       
     # Exps-related methods   

    # Exp management
    
    def _build_exps(self, experiments):
        exps = []
        for i, exp in enumerate(experiments):
            exp_id = f"{self.pr_id}_{i+1:03d}"
            exp_dict = {
                'id': exp_id,
                'hp':exp,
                'alive':True,
                'restart_epoch' : -1,
                'early_stopping_counter':0,
                'train_loss':np.array([]),
                'val_loss':np.array([]),
                'best_score': 1000000, # Setting an arbitrary high value
                'death' :'n.a.'
            }
            exps.append(exp_dict)
        return exps
         
    def _kill_exp(self, exp, reason:str):
        exp['alive'] = False
        exp['death'] = reason
        curves_path = self.path / 'learning_curves'
        np.save(curves_path / f"{exp['id']}_train_loss.npy", exp['train_loss'])
        np.save(curves_path / f"{exp['id']}_val_loss.npy", exp['val_loss']) 
        self._log_exp(exp)  
        self.n_exp_alive = self.n_exp_alive -1    
        if exp['val_loss'].shape[0] > 0 :
            print(f"      -> Killing exp {exp['id']} with a validation loss of {exp['val_loss'][-1]:.2e} | {reason}")  
        else :
            print(f"      -> Killing exp {exp['id']} | {reason}")  
                   
    def _log_exp(self, exp, log_type:str = 'train', metrics = None):
        exp_id = int(exp['id'].split('_')[-1])
        exp_log = {'pr_id':self.pr_id, 'sub_id' : exp_id, 'seq_len':self.args.seq_len, 'pred_len':self.args.pred_len, 'features':self.args.features, 
                   'lifetime': exp['train_loss'].shape[0], 'death': exp['death'], 'best_val_score':exp['best_score']}
        exp_log.update(exp['hp'])
        
        if log_type == 'test':
            exp_log.update(metrics)
            log_filepath = self.path/'testing_logs.csv'
        else :
            log_filepath = self.path/'pruning_logs.csv'
        
        exp_log = pd.DataFrame(exp_log, index = [exp['id']])
        if os.path.exists(log_filepath):
            ldf = pd.read_csv(log_filepath, index_col = 0)
            ldf = pd.concat([ldf, exp_log])
        else :
            ldf = exp_log
        ldf.to_csv(log_filepath)        

    def _setup_training_logs(self):
        
        """Initialize CSV file with headers."""
        headers = ['exp_id', 'epoch', 'train_loss', 'val_loss', 'early_stopping_counter']
        
        with open(self.train_logs_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)      

    def _resume(self):

        if os.path.exists(self.train_logs_file):
            # Resume unfinished pruning
            training_logs = pd.read_csv(self.train_logs_file, index_col=0).sort_values('epoch')
            self.last_completed_epoch = training_logs['epoch'].max()
            pruning_logs = None
            if os.path.exists(self.path/'pruning_logs.csv'):
                pruning_logs = pd.read_csv(self.path/'pruning_logs.csv', index_col=0)
                pruning_logs = pruning_logs[pruning_logs['pr_id'] == self.pr_id]
                if 'pruning' in pruning_logs['death'].unique():
                    self.n_completed_iter = (pruning_logs[pruning_logs['death'] == 'pruning']['lifetime'].max() +1) // self.pruning_epochs
                else : 
                    self.n_completed_iter = 0
            else :
                self.n_completed_iter = 0
            print(f"Found started pruning - {int(self.n_completed_iter)} pruning steps completed - resuming...")
            for exp in self.exps :
                if pruning_logs is not None :    
                    if exp['id'] in pruning_logs.index :
                        exp['death'] = pruning_logs.loc[exp['id'], 'death']
                        exp['alive'] = exp['death'] == 'n.a.'
                if exp['id'] in training_logs.index :
                    exp_trdf = training_logs.loc[exp['id']]
                    exp['restart_epoch'] = int(exp_trdf['epoch'].max())
                    if exp['restart_epoch'] == 0 :
                        exp['train_loss'] = np.array([exp_trdf['train_loss']])
                        exp['val_loss'] = np.array([exp_trdf['val_loss']]) 
                        exp['early_stopping_counter'] = int(exp_trdf['early_stopping_counter'])
                    else :
                        exp['train_loss'] = exp_trdf['train_loss'].values
                        exp['val_loss'] = exp_trdf['val_loss'].values 
                        exp['early_stopping_counter'] = exp_trdf['early_stopping_counter'].iloc[-1]
                    exp['best_score'] = exp['val_loss'].min()
                    if exp['early_stopping_counter'] >= self.args.patience:
                        exp['death'] = 'early_stopping'
                        exp['alive'] = False
                    print(f"  restoring exp {exp['id']} | last epoch : {exp['restart_epoch']}, best_score : {exp['best_score']:.2e}, early_stopping counter : {exp['early_stopping_counter']} | death : {exp['death']}")     
                    self._build_exp_model(exp)
                if not exp['alive'] :
                    self.n_exp_alive = self.n_exp_alive -1
                       
        else :
            # Initiating pruning as no training logs are found 
            self._setup_training_logs()
            self.n_completed_iter = 0
            self.last_completed_epoch = 0
        

    # Model-related methods
    
    def _build_exp_model(self, exp):
        args = copy.deepcopy(self.args)
        
        # Update args with experiment-specific parameters
        for param_name, param_value in exp['hp'].items():
            if hasattr(args, param_name):
                setattr(args, param_name, param_value)
            else:
                print(f"Warning: Parameter '{param_name}' not found in args")
        setattr(args, 'task_name', 'long_term_forecast')
        
        model = self.model_dict[args.model].Model(args).float()
        self._register_model(exp_id= exp['id'], model = model, model_args= args)
        model = model.to(self.device)
        model_optim = self._select_optimizer(model)
        scaler = None
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        return model, model_optim, scaler

    def _get_data(self, args, flag):
        data_set, data_loader = data_provider(args, flag)
        return data_set, data_loader

    def _select_optimizer(self, model):
        if self.args.optimizer == 'adamw':
            model_optim = optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.wd)         
        else :
            model_optim = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion    
         
    def _register_model(self, 
                    exp_id :str,
                    model: nn.Module, 
                    model_args = None):
        """
        Register a new model for training and tracking.
        
        Args:
            model: PyTorch model
            model_name: Human-readable name for the model
            optimizer: Associated optimizer (optional, can be set later)
            model_args: Arguments needed to instantiate the model (needed for loading)
        """
        
        model_class = model.__class__
        class_name = model_class.__name__
        module_name = model_class.__module__
                
        # Register the model
        self.model_registry[exp_id] = {
            'model_class': class_name,
            'model_module': module_name,
            'model_args': model_args,
        }
            
    def _save_model(self, 
                    model_id: str, 
                    model: nn.Module, 
                    optimizer: optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler = None,
                    reason: str = 'last'):
        """
        Save a model checkpoint, including AMP GradScaler if provided.
        
        Args:
            model_id: Unique model identifier
            model: PyTorch model
            optimizer: Associated optimizer
            scaler: Optional GradScaler for mixed precision training
            reason: Tag for the save ('last', 'best', etc.)
        """
        
        # Path operations
        model_dir = os.path.join(self.path, f"pr_{self.pr_id}", model_id)
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, f"{reason}.pt")
        
        # Get optimizer class info for restoration
        optimizer_info = None
        if optimizer is not None:
            optimizer_class = optimizer.__class__
            optimizer_info = {
                'class_name': optimizer_class.__name__,
                'module_name': optimizer_class.__module__,
                'state_dict': optimizer.state_dict(),
                # Extract constructor parameters
                'params': {
                    'lr': optimizer.param_groups[0]['lr'],
                    'weight_decay': optimizer.param_groups[0].get('weight_decay', 0),
                    'betas': optimizer.param_groups[0].get('betas', (0.9, 0.999)),
                    'eps': optimizer.param_groups[0].get('eps', 1e-8),
                    'momentum': optimizer.param_groups[0].get('momentum', 0),
                    'dampening': optimizer.param_groups[0].get('dampening', 0),
                    'nesterov': optimizer.param_groups[0].get('nesterov', False)
                }
            }
        
        # Get scaler state if provided
        scaler_state = None
        if scaler is not None:
            scaler_state = scaler.state_dict()
        
        # Save model checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_info': optimizer_info,
            'scaler_state_dict': scaler_state,  # Save scaler state
            'model_id': model_id,
            'model_class': self.model_registry[model_id]['model_class'],
            'model_module': self.model_registry[model_id]['model_module'],
            'model_args': self.model_registry[model_id]['model_args'],
            'amp_enabled': scaler is not None  # Flag to indicate if AMP was used
        }, checkpoint_path)        

    def load_model(self, 
                model_id: str, 
                reason: str = 'last', 
                load_optimizer: bool = True,
                load_scaler: bool = True):
        """
        Load a model from checkpoint without requiring a model instance.
        
        Args:
            model_id: Unique model identifier
            reason: Which save to load ('last', 'best', etc.)
            load_optimizer: Whether to load the optimizer
            load_scaler: Whether to load the AMP GradScaler
            
        Returns:
            model: Loaded model
            optimizer: Loaded optimizer (if load_optimizer is True, else None)
            scaler: Loaded GradScaler (if AMP was used and load_scaler is True, else None)
        """
        model_dir = os.path.join(self.path, f"pr_{self.pr_id}", model_id)
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"No checkpoints found for model_id: {model_id}")
        
        checkpoint_path = os.path.join(model_dir, f"{reason}.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found with reason '{reason}' for model ID {model_id}")
        
        # Load checkpoint
        # Add map_location to handle loading models trained on different devices
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Dynamically import and instantiate the model class
        module_name = checkpoint['model_module']
        class_name = checkpoint['model_class']
        model_args = checkpoint['model_args']
                
        # Import the module and get the class
        module = __import__(module_name, fromlist=[class_name])
        model_class = getattr(module, class_name)
        
        # Create model instance using saved arguments
        model = model_class(model_args)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        # Load optimizer if requested and available
        optimizer = None
        if load_optimizer and checkpoint.get('optimizer_info') is not None:
            optimizer_info = checkpoint['optimizer_info']
            
            # Dynamically import and instantiate the optimizer class
            opt_module_name = optimizer_info['module_name']
            opt_class_name = optimizer_info['class_name']
            
            # Import the optimizer module and class
            opt_module = __import__(opt_module_name, fromlist=[opt_class_name])
            optimizer_class = getattr(opt_module, opt_class_name)
            
            # Extract relevant parameters for this optimizer type
            opt_params = {}
            all_params = optimizer_info['params']
            
            # Different optimizers need different parameters
            if opt_class_name == 'SGD':
                opt_params = {
                    'lr': all_params['lr'],
                    'momentum': all_params['momentum'],
                    'dampening': all_params['dampening'],
                    'weight_decay': all_params['weight_decay'],
                    'nesterov': all_params['nesterov']
                }
            elif opt_class_name in ['Adam', 'AdamW']:
                opt_params = {
                    'lr': all_params['lr'],
                    'betas': all_params['betas'],
                    'eps': all_params['eps'],
                    'weight_decay': all_params['weight_decay']
                }
            else:
                # Default to basic params that most optimizers use
                opt_params = {
                    'lr': all_params['lr'],
                    'weight_decay': all_params['weight_decay']
                }
            
            # Create optimizer instance with correct parameters
            optimizer = optimizer_class(model.parameters(), **opt_params)
            
            # Load the optimizer state
            optimizer.load_state_dict(optimizer_info['state_dict'])
        
        # Load GradScaler if AMP was used and requested
        scaler = None
        if load_scaler and checkpoint.get('amp_enabled', False) and checkpoint.get('scaler_state_dict') is not None:
            scaler = torch.cuda.amp.GradScaler()
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return model, optimizer, scaler

    # Training utils
    
    def _log_training(self, exp, epoch):
        csv_data = [exp['id'], epoch, exp['train_loss'][-1], exp['val_loss'][-1], exp['early_stopping_counter'] ]
        
        # Write to CSV
        with open(self.train_logs_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_data)        
    
    def _training_epoch(self, model, model_optim, scaler, train_loader, criterion, epoch:int):

        train_loss = []
        model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.loss == 'FDE' :
                        loss = criterion(outputs[:,-1,:], batch_y[:,-1,:])
                    else :
                        loss = criterion(outputs, batch_y)                        
                    train_loss.append(loss.item())
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)                    
                if self.args.loss == 'FDE' :
                    loss = criterion(outputs[:,-1,:], batch_y[:,-1,:])
                else :
                    loss = criterion(outputs, batch_y)                        
                train_loss.append(loss.item())

            if self.args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()
            adjust_learning_rate_pruning(optimizer= model_optim, 
                                         max_lr = self.args.learning_rate,
                                         min_lr = self.args.learning_rate / 1000,
                                         epoch = epoch,
                                         pruning_epoch= self.pruning_epochs,
                                         n_iter = i,
                                         n_iter_per_epoch = len(train_loader),
                                         warmup_fraction= 0.25,
                                         restart_factor = 0.5
                                         ) 

        train_loss = np.average(train_loss)

        return train_loss
    
    def _validation_step(self, model, val_loader, criterion):
        total_loss = []
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                if self.args.loss == 'FDE' :
                    loss = criterion(pred[:,-1,:], true[:,-1,:])
                else :
                    loss = criterion(pred, true)
                #print(f"{pred.shape=}, {true.shape=}, {loss=}")

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        model.train()
        return total_loss
  
    def _pruning_loop(self, train_loader, val_loader, criterion):
        print("-> First step : pruning experiments ")
        start_time = time.time()
        n_pruning_iter = self.args.train_epochs // self.pruning_epochs
        last_epoch = 0 if self.n_exp_alive > 1 else self.last_completed_epoch
        for iter in range(n_pruning_iter):
            if self.n_exp_alive > 1  :
                if iter +1 > self.n_completed_iter :
                    iter_start_time = time.time()
                    scores = []
                    ids = []
                    print(f"Running iteration n° {iter+1} : epochs {iter * self.pruning_epochs } to {(iter+1) * self.pruning_epochs -1} ") 
                    for exp in self.exps :
                        show_model = True
                        if exp['alive']:
                            if iter == 0 :
                                model, model_optim, scaler = self._build_exp_model(exp)
                            else :
                                # Load last_model from checkpoint
                                model, model_optim, scaler = self.load_model(model_id= exp['id'], reason = 'last', load_optimizer=True, load_scaler=True)
                            for k in range(self.pruning_epochs):
                                epoch = self.pruning_epochs * iter + k 
                                if exp['alive'] and epoch > exp['restart_epoch']:
                                    last_epoch =max(last_epoch, epoch)
                                    try :
                                        train_loss = self._training_epoch(model, model_optim, scaler, train_loader, criterion, epoch = epoch)                                    
                                        exp['train_loss'] = np.append(exp['train_loss'], train_loss)
                                        val_loss = self._validation_step(model, val_loader, criterion)
                                        exp['val_loss'] = np.append(exp['val_loss'], val_loss)
                                        
                                        if math.isnan(val_loss) or math.isnan(train_loss) :
                                            self._kill_exp(exp, reason = 'model_divergence')                                        
                                        
                                        # EarlyStopping logic
                                        self._save_model(model_id = exp['id'], model = model, optimizer = model_optim, scaler= scaler, reason = 'last')
                                        if val_loss < exp['best_score']:
                                            # Save best_model 
                                            exp['best_score'] = val_loss
                                            self._save_model(model_id = exp['id'], model = model, optimizer = model_optim, reason = 'best')                            
                                            exp['early_stopping_counter'] = 0
                                        else :
                                            exp['early_stopping_counter'] += 1
                                        
                                        if show_model :
                                            exp_str = "  Exp id : {} | model : {:>26}".format(exp['id'],exp['hp']['model'])
                                            show_model = False
                                        else :
                                            exp_str = " " * 53 
                                        
                                        print("{} | epoch {:3} : train loss = {:.2e}, val loss = {:.2e} | earlystopping counter {}/{} | learning_rate = {:.2e} ".format(
                                                exp_str,
                                                epoch,
                                                train_loss,
                                                val_loss,
                                                exp['early_stopping_counter'],
                                                self.args.patience,
                                                model_optim.param_groups[0]['lr']
                                                ))
                                        if exp['early_stopping_counter'] >= self.args.patience :
                                            self._kill_exp(exp, reason = 'early_stopping')
                                    except :
                                        # In case the above fails, kill the exp
                                        self._kill_exp(exp, reason = 'training_failure')
                                        # Keeping the last best score, which is set very high at startup
                                        val_loss = exp['best_score']
                                        exp['train_loss'] = np.append(exp['train_loss'], exp['best_score'])
                                        exp['val_loss'] = np.append(exp['val_loss'], exp['best_score'])                                
                                    
                                    self._log_training(exp, epoch= epoch)                                
                                elif exp['alive'] and epoch <= exp['restart_epoch']:
                                    val_loss = exp['val_loss'][epoch]
                                    
                            ids.append(exp['id'])
                            scores.append(val_loss)
                    
                    scores_series = pd.Series(scores, index = ids)
                    scores_series = scores_series.sort_values(ascending = True)
                    n_exp_to_keep = max(int(scores_series.shape[0] * (1- self.pruning_factor)),1)
                    keep_thres = scores_series.iloc[n_exp_to_keep-1]
                    time_now = time.time()
                    time_from_start = time_now - start_time
                    iter_time = time_now - iter_start_time
                    time_from_start_str = f"{int(time_from_start//3600)}hr {int((time_from_start - time_from_start //3600*3600)//60)}min {int(time_from_start%60)}sec"
                    iter_time_str = f"{int(iter_time//3600)}hr {int((iter_time - iter_time //3600*3600)//60)}min {int(iter_time%60)}sec"
                    print("Iteration n° {} | epochs {} to {} | threshold = {:.2e} | Iteration time : {} - Time from start : {}".format(
                        iter + 1,
                        iter * self.pruning_epochs,
                        (iter+1) * self.pruning_epochs -1,
                        keep_thres,
                        iter_time_str,
                        time_from_start_str,
                    ))
                    for exp in self.exps :
                        if exp['val_loss'][-1] > keep_thres and exp['alive']:
                            self._kill_exp(exp, reason = 'pruning')
                
            else:
                last_alive = None
                for exp in self.exps :
                    if exp['alive']:
                        last_alive = exp
                return last_alive, last_epoch, start_time
                
    def _last_exp_training_loop(self, last_alive_exp, last_epoch,train_loader, val_loader, criterion, start_time):
        n_remaining_epochs = self.args.train_epochs - last_epoch
        if n_remaining_epochs > 0 and last_alive_exp is not None :
            model, model_optim, scaler = self.load_model(model_id= last_alive_exp['id'], reason = 'last', load_optimizer=True, load_scaler=True)

            print("-> Second step : finishing training of last alive experiment (id: {}, model: {})".format(last_alive_exp['id'], last_alive_exp['hp']['model']))

            for k in range(n_remaining_epochs):
                epoch_start_time = time.time()
                epoch = last_epoch + k +1 
                if last_alive_exp['alive'] and epoch > last_alive_exp['restart_epoch'] :
                    train_loss = self._training_epoch(model, model_optim, scaler, train_loader, criterion, epoch = epoch)
                    last_alive_exp['train_loss'] = np.append(last_alive_exp['train_loss'], train_loss)
                    val_loss = self._validation_step(model, val_loader, criterion)
                    last_alive_exp['val_loss'] = np.append(last_alive_exp['val_loss'], val_loss)
                    
                    # EarlyStopping logic
                    self._save_model(model_id = last_alive_exp['id'], model = model, optimizer = model_optim, scaler= scaler, reason = 'last')
                    if val_loss < last_alive_exp['best_score']:
                        # Save best_model 
                        last_alive_exp['best_score'] = val_loss
                        self._save_model(model_id = last_alive_exp['id'], model = model, optimizer = model_optim, reason = 'best')                            
                        last_alive_exp['early_stopping_counter'] = 0
                    else :
                        last_alive_exp['early_stopping_counter'] += 1
                    
                    self._log_training(last_alive_exp, epoch= epoch)
                    
                    time_now = time.time()
                    time_from_start = time_now - start_time
                    epoch_time = time_now - epoch_start_time
                    time_from_start_str = f"{int(time_from_start//3600)}hr {int((time_from_start - time_from_start //3600*3600)//60)}min {int(time_from_start%60)}sec"
                    epoch_time_str = f"{int(epoch_time//3600)}hr {int((epoch_time - epoch_time //3600*3600)//60)}min {int(epoch_time%60)}sec"                    
                    print("  epoch {:3} : train loss = {:.2e}, val loss = {:.2e} | earlystopping counter {}/{} | learning_rate = {:.2e} | Epoch time : {} - Time from start : {}".format(
                        epoch,
                        train_loss,
                        val_loss,
                        last_alive_exp['early_stopping_counter'],
                        self.args.patience,
                        model_optim.param_groups[0]['lr'],
                        epoch_time_str,
                        time_from_start_str
                    ))
                    if last_alive_exp['early_stopping_counter'] >= self.args.patience :
                        self._kill_exp(last_alive_exp, reason = 'early_stopping')                    
            
    # Main training methods

    def train(self, setting=None):

        print(">>> TRAINING <<<")
        train_data, train_loader = self._get_data(self.args, flag='train')
        val_data, val_loader = self._get_data(self.args, flag='val')
        criterion = self._select_criterion()
       
        last_alive_exp, last_epoch, start_time = self._pruning_loop(train_loader=train_loader, val_loader=val_loader, criterion=criterion)
        self._last_exp_training_loop(last_alive_exp=last_alive_exp,
                                     last_epoch=last_epoch,
                                     train_loader=train_loader,
                                     val_loader=val_loader,
                                     criterion=criterion,
                                     start_time=start_time)
        # Logging the training of alive exps
        for exp in self.exps :
            if exp['alive']:
                self._kill_exp(exp, reason = 'epoch_budget')
                
        print("-> Training completed")
                        
    def test(self, setting, test_pruned_exps:bool=False):
        
        print(">>> TESTING <<<")
        test_data, test_loader = self._get_data(self.args, flag='test')
 
        for exp in self.exps:
            test_exp = True
            # Selecting exps to test
            if not test_pruned_exps :
                if exp['death'] == 'pruning' or exp['death'] == 'training_failure' or exp['death'] == 'model_divergence':
                    test_exp = False

            if test_exp :
                # Load best_model from checkpoint
                model, _, _ = self.load_model(model_id= exp['id'], reason = 'best', load_optimizer=False, load_scaler=False)

                preds = []
                trues = []
                folder_path = self.path / exp['id']  
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                model.eval()
                with torch.no_grad():
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                        # encoder - decoder
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                        outputs = outputs.detach().cpu().numpy()
                        batch_y = batch_y.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            outputs_shape = outputs.shape
                            batch_y_shape = batch_y.shape
                            outputs = test_data.inverse_transform(outputs.reshape(outputs_shape[0] * outputs_shape[1], -1)).reshape(outputs_shape)
                            batch_y = test_data.inverse_transform(batch_y.reshape(batch_y_shape[0] * batch_y_shape[1], -1)).reshape(batch_y_shape)
                
                        outputs = outputs[:, :, f_dim:]
                        batch_y = batch_y[:, :, f_dim:]

                        pred = outputs
                        true = batch_y

                        preds.append(pred)
                        trues.append(true)
                        if i % 20 == 0:
                            input = batch_x.detach().cpu().numpy()
                            if test_data.scale and self.args.inverse:
                                shape = input.shape
                                input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                            gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                            pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                            visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                preds = np.concatenate(preds, axis=0)
                trues = np.concatenate(trues, axis=0)
                preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                    
                mae, mse, rmse, mape, mspe, fde = metric(preds, trues)
                metrics = {'mae':mae, 'mse':mse, 'mape':mape, 'mspe':mspe, 'rmse':rmse, 'fde':fde}
                # print('fde:{:.4e}, rmse:{:.4e}, mse:{:.4e}, mae:{:.4e}'.format(fde,rmse,mse, mae))
                f = open("result_long_term_forecast.txt", 'a')
                f.write(exp['id'] + "  \n")
                f.write('mse:{}, mae:{}'.format(mse, mae))
                f.write('\n')
                f.write('\n')
                f.close()
                
                print("  Exp id : {:6} | model : {:26} | RMSE : {:.2e} | FDE {:.2e} ".format(
                    exp['id'],
                    exp['hp']['model'],
                    rmse,
                    fde))
                np.save(folder_path / 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
                np.save(folder_path / 'pred.npy', preds)
                np.save(folder_path / 'true.npy', trues)
                
                self._log_exp(exp, log_type= 'test' , metrics=metrics)
                # log(metrics=metrics, args = self.args)
                
    def test_selected_experiments(self, exps_df_path = './pruning/best_exps.csv'):
        edf = pd.read_csv(exps_df_path, index_col = 0)
        log_filepath = self.path/'top_exps'/'results.csv'
        if os.path.exists(log_filepath):
            ldf = pd.read_csv(log_filepath, index_col = 0)
        else :
            ldf = None
        test_args = copy.deepcopy(self.args)
        setattr(test_args, 'task_name', 'long_term_forecast')                    
        for s in edf['seq_len'].unique():
            s_edf = edf[edf['seq_len']==s]
            setattr(test_args, 'seq_len', s)
            setattr(test_args, 'label_len', int(s//2))
            for p in s_edf['pred_len'].unique() :
                setattr(test_args, 'pred_len', p)
                for f in s_edf['features'].unique():
                    setattr(test_args, 'features', f)                
                    print(f"  Processing time horizon {s}-{p} - {f}")
                    thdf = s_edf[(s_edf['pred_len'] == p) & (s_edf['features'] == f)]
                    test_data, test_loader = self._get_data(test_args, flag='test')
                    for model_id, model_name in zip(thdf.index, thdf['model']):
                        setattr(test_args, 'model', model_name)
                        if ldf is not None and model_id in ldf.index :
                            # Skipping the model as it is already registrered
                            print(f"    Skipping testing of {model_id} : already present in logs.")
                            model = None
                        else :
                            # Loading model 
                            model_dir = Path(self.path) / 'top_exps' / model_id
                            checkpoint_path = os.path.join(model_dir, "best.pt")
                            if not os.path.exists(checkpoint_path):
                                raise FileNotFoundError(f"No checkpoint found for model ID {model_id}")
                            ### Load checkpoint and map_location to handle loading models trained on different devices
                            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                            ### Dynamically import and instantiate the model class
                            module_name = checkpoint['model_module']
                            class_name = checkpoint['model_class']
                            model_args = checkpoint['model_args']
                            ### Import the module and get the class
                            module = __import__(module_name, fromlist=[class_name])
                            model_class = getattr(module, class_name)
                            ### Create model instance using saved arguments
                            model = model_class(model_args)
                            try :
                                model.load_state_dict(checkpoint['model_state_dict'])
                                model = model.to(self.device)
                            except :
                                model = None
                                print(f"    -> ERROR : could not manage to load model {model_name} for exp {model_id}")       
                        
                        if model is not None :
                            # Testing routine   
                            preds = []
                            trues = []
                            model.eval()
                            with torch.no_grad():
                                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                                    batch_x = batch_x.float().to(self.device)
                                    batch_y = batch_y.float().to(self.device)
                                    batch_x_mark = batch_x_mark.float().to(self.device)
                                    batch_y_mark = batch_y_mark.float().to(self.device)

                                    # decoder input
                                    dec_inp = torch.zeros_like(batch_y[:, -test_args.pred_len:, :]).float()
                                    dec_inp = torch.cat([batch_y[:, :test_args.label_len, :], dec_inp], dim=1).float().to(self.device)
                                    # encoder - decoder
                                    if test_args.use_amp:
                                        with torch.cuda.amp.autocast():
                                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                    else:
                                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                                    f_dim = -1 if test_args.features == 'MS' else 0
                                    outputs = outputs[:, -test_args.pred_len:, :]
                                    batch_y = batch_y[:, -test_args.pred_len:, :].to(self.device)
                                    outputs = outputs.detach().cpu().numpy()
                                    batch_y = batch_y.detach().cpu().numpy()
                                    if test_data.scale and test_args.inverse:
                                        outputs_shape = outputs.shape
                                        batch_y_shape = batch_y.shape
                                        outputs = test_data.inverse_transform(outputs.reshape(outputs_shape[0] * outputs_shape[1], -1)).reshape(outputs_shape)
                                        batch_y = test_data.inverse_transform(batch_y.reshape(batch_y_shape[0] * batch_y_shape[1], -1)).reshape(batch_y_shape)
                            
                                    outputs = outputs[:, :, f_dim:]
                                    batch_y = batch_y[:, :, f_dim:]

                                    pred = outputs
                                    true = batch_y

                                    preds.append(pred)
                                    trues.append(true)
                                    if i % 20 == 0:
                                        input = batch_x.detach().cpu().numpy()
                                        if test_data.scale and test_args.inverse:
                                            shape = input.shape
                                            input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                                        pred = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                                        visual(gt, pred, os.path.join(model_dir, str(i) + '.pdf'))

                            preds = np.concatenate(preds, axis=0)
                            trues = np.concatenate(trues, axis=0)
                            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                                
                            mae, mse, rmse, mape, mspe, fde = metric(preds, trues)
                            metrics = {'mae':mae, 'mse':mse, 'mape':mape, 'mspe':mspe, 'rmse':rmse, 'fde':fde}
                            
                            print("    Exp id : {:6} | model : {:26} | RMSE : {:.2e} | FDE {:.2e} ".format(
                                model_id,
                                model_name,
                                rmse,
                                fde))
                            np.save(model_dir / 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
                            np.save(model_dir / 'pred.npy', preds)
                            np.save(model_dir / 'true.npy', trues)
                                                
                            exp_log = pd.DataFrame(metrics, index = [model_id])
                            if ldf is not None:
                                ldf = pd.concat([ldf, exp_log])
                            else :
                                ldf = exp_log
                            ldf.to_csv(log_filepath)                                      
                                                    
                
        