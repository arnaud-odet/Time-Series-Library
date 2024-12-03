import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import datetime
import matplotlib.pyplot as plt


# Utils

class CustomEarlyStopping:
    
    def __init__(self, patience=5, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_model_weights = None

    def __call__(self, val_loss, model, epoch, n_epochs, train_loss):
        score = -val_loss

        post_message = ''

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch+1
            self.train_loss_best_epoch = train_loss
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                # Restore the best model weights if early stopping is triggered
                if -self.best_score < 0.001 :
                    post_message += f"\nRestoring best model weights from epoch {self.best_epoch} with Validation Loss = {-self.best_score:.4e}"
                else :
                    post_message += f"\nRestoring best model weights from epoch {self.best_epoch} with Validation Loss = {-self.best_score:.4f}"
                model.load_state_dict(self.best_model_weights)

        else:
            self.best_score = score
            self.best_epoch = epoch+1
            self.train_loss_best_epoch = train_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        message = f"Epoch {epoch+1}/{n_epochs} - Training Loss: {train_loss:.4e}, Validation Loss: {val_loss:.4e} - Early Stopping counter: {self.counter}/{self.patience}"

        return message + post_message

    def save_checkpoint(self, val_loss, model):
        """Saves the model when the validation loss decreases."""
        # Save the current best model's state_dict
        self.best_model_weights = model.state_dict().copy()
        self.val_loss_min = val_loss

class LayerTracker:
    
    def __init__(self, main_model):
        self.reset()
        self.main_model = main_model
        self.hooks = []
        self.register_hooks()

    def reset(self):
        self.layers = []
        self.seen = set()
        if hasattr(self, 'hooks'):
            for hook in self.hooks:
                hook.remove()
            self.hooks = []

    def _is_inside_transformer(self, module):
        """Check if the module is inside a TransformerEncoder"""
        current = module
        for name, m in self.main_model.named_modules():
            if m is current:
                # Check if any parent in the name path is a transformer
                parent_names = name.split('.')
                for parent_name in parent_names[:-1]:  # Exclude the current module name
                    parent = dict(self.main_model.named_modules()).get(parent_name)
                    if isinstance(parent, nn.TransformerEncoder):
                        return True
        return False

    def hook(self, module, input, output):
        # Skip if we've seen this module
        if module in self.seen:
            return
        # Add the module if it is not a TransformerEncoder and not inside a TransformerEncoder, not to duplicate LayerNorm and Dropout
        if ((isinstance(module, nn.TransformerEncoderLayer) or not self._is_inside_transformer(module)) and not isinstance(module, nn.TransformerEncoder) ):
            self.seen.add(module)
            self.layers.append(module)

    def register_hooks(self):
        self.reset()
        for name, module in self.main_model.named_modules():
            if module != self.main_model:
                hook = module.register_forward_hook(self.hook)
                self.hooks.append(hook)

def get_nb_units(layer):
    """Get number of units/channels/features for different layer types"""
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        return layer.out_channels
    elif isinstance(layer, nn.Linear):
        return layer.out_features
    elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
        return layer.num_features
    
    # Recurrent layers
    elif isinstance(layer, nn.LSTM):
        return layer.hidden_size
    elif isinstance(layer, nn.GRU):
        return layer.hidden_size
    
    # Transformer components
    elif isinstance(layer, nn.TransformerEncoderLayer):
        return layer.self_attn.embed_dim
    elif isinstance(layer, nn.TransformerDecoderLayer):
        return layer.self_attn.embed_dim
    elif isinstance(layer, nn.MultiheadAttention):
        return layer.embed_dim
    elif isinstance(layer, nn.Embedding):
        return layer.embedding_dim
    
    # Layers without units in the traditional sense
    elif isinstance(layer, (nn.ReLU, nn.MaxPool2d, nn.Dropout, 
                          nn.LayerNorm, nn.Softmax, nn.Tanh)):
        return 0
    else:
        return 0

def show_models_layers(model):
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")


# Wraper

class ModelWrapper():
    
    def __init__(self, model ,args):
        self.model = model
        self.history = {'train_loss':[], 'val_loss' : []}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model.device = self.device
        self.model.to(self.model.device)
        self.args = args
        
        # Summary args  
        self.skip_connections = args.skip_co
        self.transformer_num_heads = args.n_heads
        self.transformer_ff_units = args.d_ff
        self.dropout_rate = args.dropout
        self.input_shape = (1, args.seq_len, args.enc_in)
        
    def summary(self, pad_1:int=30, pad_2:int=18):
        layers_display = {}
        layer_types = {}
        nb_params = 0
        
        tracker = LayerTracker(self.model)
        # Run a forward pass with dummy inputs
        dummy_input = torch.zeros(self.input_shape).to(self.device)
        with torch.no_grad():
            dummy_output = self.model(dummy_input, None, None, None)
        for layer in tracker.layers :
            if not isinstance(layer, type(self.model)):
                layer_type = str(type(layer)).split('.')[-1].strip('>')[:-1]
                if layer_type not in layer_types :
                    layer_types[layer_type] = 0
                else :
                    layer_types[layer_type] += 1
                layer_tr_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                layer_nontr_params = sum(p.numel() for p in layer.parameters() if not p.requires_grad)
                layer_nontr_params += sum(b.numel() for b in layer.buffers())
                ld_name = layer_type + '_' + str(layer_types[layer_type])
                if layer_type != 'Dropout':
                    layers_display[ld_name] = {'n_unit' : get_nb_units(layer), 'n_tr_param' : layer_tr_params,'n_nontr_param' : layer_nontr_params}
                else :
                    pass # Not displaying dropout
                nb_params += layer_tr_params
                nb_params += layer_nontr_params
         
        print("Model quick summary")
        print("="*(pad_1 + 3 * pad_2 +3))
        print('Layer name' + (pad_1 - len('Layer name')) * ' ' + '|' +(pad_2 - len('Nb units')) * ' ' + 'Nb units'+ '|' +(pad_2 - len('Nb tr params')) * ' ' + 'Nb tr params'+ '|' +(pad_2 - len('Nb non tr params')) * ' ' + 'Nb non tr params')
        for k,v in layers_display.items() :
            str_n_unit = f"{v['n_unit']:_}".replace("_"," ") if v['n_unit'] > 0 else '-'
            str_n_tr = f"{v['n_tr_param']:_}".replace("_"," ")
            str_n_ntr = f"{v['n_nontr_param']:_}".replace("_"," ")
            line = k + (pad_1 - len(k)) * ' ' + '|' + (pad_2 - len(str_n_unit)) * ' ' + str_n_unit+ '|' +(pad_2 - len(str_n_tr)) * ' ' + str_n_tr+ '|' + (pad_2 - len(str_n_ntr)) * ' ' + str_n_ntr
            print(line)
        print("="*(pad_1 + 3 * pad_2 +3))
        if hasattr(self, 'skip_connections') :
            print(f"Skip_connections : {self.skip_connections}")
        if hasattr(self, 'transformer_num_heads') :
            print(f"Transformer_nb_heads : {self.transformer_num_heads}")
        if hasattr(self, 'transformer_ff_units') :
            print(f"Transformer_FF_units : {self.transformer_ff_units}")
        if hasattr(self, 'dropout_rate') :        
            print(f"Dropout_rate : p = {self.dropout_rate}")
        trainable_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_param_count = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        non_trainable_param_count += sum(b.numel() for b in self.model.buffers())
        print(f"\nNb trainable parameters : {trainable_param_count:_}".replace("_"," "))
        print(f'Nb non-trainable parameters : {non_trainable_param_count:_}'.replace("_"," "))
        print(f"Total nb parameters : {trainable_param_count+non_trainable_param_count:_}".replace("_"," ")) 
        if nb_params != trainable_param_count + non_trainable_param_count:
            print(f"WARNING : Numbers of parameters do not match between detail and total, you may want to check") 
        self.trainable_params = trainable_param_count
        self.non_trainable_params = non_trainable_param_count   
        print(f"\nDummy input shape : {dummy_input.shape} | dummy output shape : {dummy_output.shape}")     

    def fit(self, train_loader, val_loader, args, verbose:bool = True):
        
        early_stopping = CustomEarlyStopping(patience=args.patience)
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, betas = args.betas, weight_decay=args.weight_decay)
        n_batchs = len(train_loader)
        n_samples = (n_batchs * train_loader.batch_size) if train_loader.drop_last else len(train_loader.dataset)
        message = f'Fit in progress ...'
        
        scaler = GradScaler()
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
                                                steps_per_epoch=n_batchs, epochs=args.train_epochs)
        
        # Training loop
        for epoch in range(args.train_epochs):
            self.model.train()  
            train_loss = 0.0
            # Training loop
            for i, (inputs, targets) in enumerate(train_loader):
                if verbose :
                    print(message + f" | Epoch {epoch+1}/{args.train_epochs} : batch {i+1}/{n_batchs}" + 20*' ', end='\r')
                inputs, targets = inputs.to(next(self.model.parameters()).device), targets.to(next(self.model.parameters()).device)
                # Zero the gradients
                optimizer.zero_grad()
                # Autocasting (mixed precision)
                with autocast():
                    # Forward pass
                    outputs = self.model(inputs, x_mark_enc = None, x_dec=None, x_mark_dec = None) ### Revoir
                    # Compute the loss
                    loss = criterion(outputs, targets)
                # Backward pass and optimization step through scaler object
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item() 

            train_loss = train_loss / n_samples
            
            # Validation step
            val_loss = 0.0
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(next(self.model.parameters()).device), val_targets.to(next(self.model.parameters()).device)
                    with autocast():
                        # Forward pass
                        val_outputs = self.model(val_inputs, x_mark_enc = None, x_dec=None, x_mark_dec = None) ### Revoir
                        val_loss += criterion(val_outputs, val_targets).item()
                val_loss = val_loss / len(val_loader.dataset)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)               
                        
            # Early stopping logic
            message = early_stopping(val_loss, model=self.model, epoch=epoch, n_epochs=args.train_epochs, train_loss = train_loss)
            if early_stopping.early_stop:
                if verbose :
                    print('\n' +message, end = '\n')
                # Recording fit arguments 
                self.batch_size = train_loader.batch_size
                self.learning_rate = args.learning_rate
                self.betas = args.betas
                self.weight_decay = args.weight_decay
                self.max_epochs = args.train_epochs
                self.patience = args.patience
                self.n_epochs = early_stopping.best_epoch +1
                self.val_loss = - early_stopping.best_score
                self.train_loss = early_stopping.train_loss_best_epoch
                if args.log :
                    self.log()
                break
            
            # Stopping logic 
            if epoch +1 == args.train_epochs :
                self.model.load_state_dict(early_stopping.best_model_weights)
                if verbose : 
                    print('\n' +message, end = '\n')
                    if -early_stopping.best_score < 0.001 :
                        print(f"Restoring best model weights from epoch {early_stopping.best_epoch} with Validation Loss = {-early_stopping.best_score:.4e}")
                    else :
                        print(f"Restoring best model weights from epoch {early_stopping.best_epoch} with Validation Loss = {-early_stopping.best_score:.4f}")               
                if args.log :
                    self.log()
       
    def predict(self, X_test, batch_size = 64):        
        self.model.eval() 
        
        # If X_test is a numpy array, convert it to a PyTorch tensor
        if isinstance(X_test, np.ndarray):
            X_test = torch.tensor(X_test, dtype=torch.float32)

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(self.device)
                outputs = self.model(batch, x_mark_enc = None, x_dec=None, x_mark_dec = None) ### Revoir
                predictions.append(outputs.cpu())
    
        return torch.cat(predictions).numpy()       
                    
    def log(self): #TDO
        pass
    
    def overview(self):
        print(f"Model : {self.args.model}")
        trainable_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_param_count = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        non_trainable_param_count += sum(b.numel() for b in self.model.buffers())
        print(f"Nb trainable parameters : {trainable_param_count:_}".replace("_"," "))
        print(f'Nb non-trainable parameters : {non_trainable_param_count:_}'.replace("_"," "))
        print(f"Total nb parameters : {trainable_param_count+non_trainable_param_count:_}".replace("_"," ")) 
        self.trainable_params = trainable_param_count
        self.non_trainable_params = non_trainable_param_count
        
    def print_layers(self):
        show_models_layers(self.model)