from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from utils.log import log
import math
from pathlib import Path
import pandas as pd

warnings.filterwarnings('ignore')


class Pruning(Exp_Basic):
    def __init__(self, 
                args,
                experiments:list,
                pruning_epochs: int=2,
                pruning_factor: float = 0.5,
                ):
        """
        Run pruning over the provided exps.
        Exp must be passed as a list of dict where each dict keys contains model hyperparameters of the corresponding experiment.
        """
        super(Pruning, self).__init__(args)
        try :
            pruning_directory = Path(args.pruning_directory)
        except :
            pruning_directory = Path('./pruning')
        if not os.path.exists(pruning_directory):
            os.mkdir(pruning_directory)
            os.mkdir(pruning_directory / 'learning_curves')
        self.path = pruning_directory
        self.pruning_epochs = pruning_epochs
        self.pruning_factor = pruning_factor
        if os.path.exists(pruning_directory/'pruning_logs.csv'):
            ldf = pd.read_csv(pruning_directory/'pruning_logs.csv', index_col = 0)
            self.pr_id = ldf['pruning_id'].max()+1
        else :
            self.pr_id = 1
        self.exps = self._build_exps(experiments)
       
     # Exps-related methods   
    
    def _build_exps(self, experiments):
        exps = []
        for i, exp in enumerate(experiments):
            exp_id = f"{self.pr_id}_{i+1}"
            exp_dict = {
                'id': exp_id,
                'hp':exp,
                'alive':True,
                'early_stopping_counter':0,
                'train_loss':[],
                'val_loss':[],
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
        print(f"    -> Killing exp {exp['id']} with a score of {exp['values'][-1]:.2e} | {reason}")  
              
    def _log_exp(self,exp):
        exp_log = {'pr_id':self.pr_id, 'lifetime': exp['train_loss'].shape[0], 'death': exp['death'], 'best_score':exp['best_score']}
        if os.path.exists(self.path/'pruning_logs.csv'):
            ldf = pd.read_csv(self.path/'pruning_logs.csv')
            ldf.loc[exp['id']] = exp_log
        else :
            ldf = pd.DataFrame(exp_log, index = exp['id'])
        ldf.to_csv(self.path/'pruning_logs.csv')
    
    # Model-related methods
    
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optimizer == 'adamw':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.wd)         
        else :
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion    
         
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
        model_dir = os.path.join(self.path, model_id)
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
        model_dir = os.path.join(self.path, model_id)
        
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
        model = model_class(**model_args)
        model.load_state_dict(checkpoint['model_state_dict'])
        
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

    # Training 
    
    def _training_epoch(self, model, model_optim, scaler, train_loader, criterion):

        train_loss = []
        model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
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

        train_loss = np.average(train_loss)
        adjust_learning_rate(model_optim, epoch + 1, self.args) ## REVOIR EPOCH ICI 

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
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
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
        self.model.train()
        return total_loss
    
    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        criterion = self._select_criterion()
       
        n_iter = self.args.train_epochs // self.pruning_factor
        for iter in range(n_iter):
            scores = []
            ids = []
            for exp in self.exps :
                if exp['alive']:
                    if iter == 0 :
                        # Customize args
                        
                        #################
                        ##### To do #####
                        #################
                        
                        # Load model from args
                        model = self._build_model()
                        model_optim = self._select_optimizer()
                        scaler = None
                        if self.args.use_amp:
                            scaler = torch.cuda.amp.GradScaler()
                        
                    else :
                        # Load last_model from checkpoint
                        model, model_optim, scaler = self.load_model(model_id= exp['id'], reason = 'last', load_optimizer=True, load_scaler=True)
                    for k in range(self.pruning_epochs):
                        train_loss = self._training_epoch(model, model_optim, scaler, train_loader, criterion)
                        exp['train_loss'] = np.append(exp['train_loss'], train_loss)
                        val_loss = self._validation_step(model, val_loader, criterion)
                        exp['val_loss'] = np.append(exp['val_loss'], val_loss)
                        
                        # EarlyStopping logic
                        self._save_model(model_id = exp['id'], model = model, optimizer = exp['model_optim'], reason = 'last')
                        if exp['val_loss'][-1] < exp['best_score']:
                            # Save best_model 
                            self._save_model(model_id = exp['id'], model = model, optimizer = exp['model_optim'], reason = 'best')                            
                            exp['early_stopping_counter'] = 0
                        else :
                            exp['early_stopping_counter'] += 1
                            if exp['early_stopping_counter'] >= self.args.patience :
                                self._kill_exp(exp, reason = 'EarlyStopping')
                                 
                    ids.append[exp['id']]
                    scores.append(val_loss)
            scores_series = pd.Series(scores, index = ids)
            scores_series = scores_series.sort_values(ascending = True)
            n_exp_to_keep = int(scores_series.shape[0] * self.pruning_factor)
            keep_thres = scores_series.iloc[n_exp_to_keep]
            print(f"Iter n° {iter + 1} - threshold = {keep_thres:.2e}")
            for exp in self.exps :
                if exp['values'][-1] < keep_thres and exp['alive']:
                    self._kill_exp(exp, reason = 'Pruning')
        
        pass
    
    def test(self, test_only_surviving_exp:bool=True):
        
        test_data, test_loader = self._get_data(flag='test')
 
        for exp in self.exps:
            test_exp = False
            # Selecting exps to test
            if test_only_surviving_exp :
                if exp['alive']:
                    test_exp = True
            else :
                test_exp = True

            if test_exp :
                # Load best_model from checkpoint
                model, _, _ = self.load_model(model_id= exp['id'], reason = 'last', load_optimizer=False, load_scaler=False)

                preds = []
                trues = []
                folder_path = self.args.results_path + setting + '/' ### A revoir
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
                #print('test shape:', preds.shape, trues.shape)
                preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                #print('test shape:', preds.shape, trues.shape)

                
                # dtw calculation
                if self.args.use_dtw:
                    dtw_list = []
                    manhattan_distance = lambda x, y: np.abs(x - y)
                    for i in range(preds.shape[0]):
                        x = preds[i].reshape(-1,1)
                        y = trues[i].reshape(-1,1)
                        if i % 100 == 0:
                            print("calculating dtw iter:", i)
                        d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                        dtw_list.append(d)
                    dtw = np.array(dtw_list).mean()
                else:
                    dtw = 'not calculated'
                    

                mae, mse, rmse, mape, mspe, fde = metric(preds, trues)
                print('fde:{:.4e}, rmse:{:.4e}, mse:{:.4e}, mae:{:.4e}, dtw:{}'.format(fde,rmse,mse, mae, dtw))
                f = open("result_long_term_forecast.txt", 'a')
                f.write(setting + "  \n")
                f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
                f.write('\n')
                f.write('\n')
                f.close()

                np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
                np.save(folder_path + 'pred.npy', preds)
                np.save(folder_path + 'true.npy', trues)
                
                # logging
                trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                non_trainable_param_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
                non_trainable_param_count += sum(b.numel() for b in model.buffers())
                metrics = {'nb_params':trainable_param_count+ non_trainable_param_count, 
                        'nb_tr_params':trainable_param_count,
                        'nb_nontr_params':non_trainable_param_count,
                        'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mspe': mspe, 'fde':fde}
                
                log(metrics=metrics, args = self.args)
                
