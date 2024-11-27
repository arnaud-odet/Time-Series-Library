import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


 
class LSTransformer(nn.Module):
    
    def __init__(self, configs):
        
        super().__init__()
        
        # Model Architecture
        self.skip_connections = configs.skip_co        
        self.dropout = nn.Dropout(configs.dropout)
        self.layer_norm_module_list = nn.ModuleList()
        self.skip_connections_module = nn.ModuleList()

        ## Step 1: LSTM layer to embed the past trajectories 
        self.lstm = nn.ModuleList()
        lstm_previous_size = configs.enc_in
        for i in range(configs.lstm_layers):
            self.lstm.append(nn.LSTM(lstm_previous_size, configs.d_lstm, batch_first=True))
            if self.skip_connections:
                self.skip_connections_module.append(nn.Linear(lstm_previous_size,configs.d_lstm))
            lstm_previous_size = configs.d_lstm
            if self.layer_norm :
                self.layer_norm_module_list.append(nn.LayerNorm(configs.d_lstm))  

        self.projection = nn.Linear(configs.d_lstm, configs.d_model)
        
        ## Step 2: Transformer multi-head attention layer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=configs.d_model, 
                                       nhead=configs.n_heads,
                                       dim_feedforward = configs.d_ff,
                                       dropout= configs.dropout, 
                                       batch_first=True),
                                       num_layers=configs.e_layers)
        
        ## Step 3: Feed-forward network to predict future trajectories
        self.linear_layers = nn.ModuleList()  
        ff_previous_size = configs.d_model
        for i in range(configs.d_layers):
            self.linear_layers.append(nn.Linear(ff_previous_size, configs.d_fc))            
            ff_previous_size = configs.d_fc  
            if self.layer_norm :
                self.layer_norm_module_list.append(nn.LayerNorm(configs.d_fc))        
        
        ## Step 4: final output layer
        self.output_layer = nn.Linear(configs.d_fc, configs.c_out)

        ## Step 5: Sending the model to the device 
        self.to(self.device)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        # x shape: (batch_size, input_len, n_features)
        batch_size = x_enc.shape[0]
        # Step 1: LSTM
        for i,lstm in enumerate(self.lstm) :
            residuals = x_enc
            x_enc, _ = lstm(x_enc)
            if self.skip_connections :
                x_enc = x_enc + self.skip_connections_module[i](residuals)
            if self.layer_norm:
                x_enc = self.layer_norm_module_list[i](x_enc)
            x_enc = self.dropout(x_enc)                 
        x_enc = self.projection(x_enc)         

        # Step 2: Transformer
        x_enc = self.transformer(x_enc)  # Shape: (batch_size, transformer_embed_dim)

        # Step 3: Fully connected layers       
        for i,linear in enumerate(self.linear_layers):
            residuals = x_enc  
            x_enc = linear(x_enc) 
            if residuals.shape[-1] == x_enc.shape[-1] and self.skip_connections: # Only adding residuals if dimensions match
                x_enc = x_enc + residuals
            if self.layer_norm :
                x_enc = self.layer_norm_module_list[i+len(self.lstm)](x_enc)
            x_enc = self.dropout(x_enc)  
            x_enc = F.relu(x_enc) 
        x_enc = self.output_layer(x_enc)  # Shape: (batch_size, output_len * n_output_feature)
        x_enc = x_enc.view(batch_size, self.output_len, self.output_dim)     
        return x_enc
   
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None