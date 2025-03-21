import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Reshape to [1, max_len, 1, d_model]
        pe = pe.unsqueeze(0).unsqueeze(2)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, n_agents, features]
        """
        # Add a linear layer to project input to d_model dimensions if needed
        if not hasattr(self, 'project'):
            self.project = nn.Linear(x.size(-1), self.d_model).to(x.device)
        
        # Project input to d_model dimensions
        x = self.project(x)
        
        # Add positional encoding
        return self.dropout(x + self.pe[:, :x.size(1), :, :])



class SetAttentionBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim_in, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim_in, dim_in * 4),
            nn.ReLU(),
            nn.Linear(dim_in * 4, dim_out)
        )
        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention
        x1 = self.norm1(x)
        attention_output, _ = self.mha(x1, x1, x1, key_padding_mask=mask)
        x = x + self.dropout(attention_output)
        
        # Feed forward
        x2 = self.norm2(x)
        ff_output = self.ff(x2)
        x = x + self.dropout(ff_output)
        
        return x

class SAB_Based_Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(configs.d_model, configs.dropout)
        
        # Create multiple encoder layers
        self.encoder_layers = nn.ModuleList()
        for _ in range(configs.e_layers):
            layer = nn.ModuleDict({
                # Two temporal SABs per layer
                'temporal_sab1': SetAttentionBlock(configs.d_model, configs.d_model, configs.n_heads, configs.dropout),
                'temporal_sab2': SetAttentionBlock(configs.d_model, configs.d_model, configs.n_heads, configs.dropout),
                # One social SAB per layer
                'social_sab': SetAttentionBlock(configs.d_model, configs.d_model, configs.n_heads, configs.dropout)
            })
            self.encoder_layers.append(layer)

    def forward(self, x):
        # x shape: [batch_size, seq_len, n_agents, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply each encoder layer sequentially
        for layer in self.encoder_layers:
            # Apply temporal SABs (operating on sequence dimension)
            batch_size, seq_len, n_agents, d_model = x.shape
            x_temporal = x.reshape(batch_size * n_agents, seq_len, d_model)
            x_temporal = layer['temporal_sab1'](x_temporal)
            x_temporal = layer['temporal_sab2'](x_temporal)
            x = x_temporal.reshape(batch_size, n_agents, seq_len, d_model)
            
            # Apply social SAB (operating on agents dimension)
            x = x.transpose(1, 2)  # [batch_size, seq_len, n_agents, d_model]
            x_social = x.reshape(batch_size * seq_len, n_agents, d_model)
            x_social = layer['social_sab'](x_social)
            x = x_social.reshape(batch_size, seq_len, n_agents, d_model)
        
        return x



 
class Model(nn.Module):
    
    def __init__(self, configs):
        
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        
        # Model Architecture
        self.skip_connections = configs.skip_co        
        self.layer_norm = configs.use_norm
        self.dropout = nn.Dropout(configs.dropout)
        self.layer_norm_module_list = nn.ModuleList()
        self.skip_connections_module = nn.ModuleList()
        
        ## Step 1: SAB_based encoder
        self.encoder = SAB_Based_Encoder(configs)
        
        ## Step 2: Feed-forward network to predict future trajectories
        self.linear_layers = nn.ModuleList()  
        ff_previous_size = configs.d_model
        for i in range(configs.d_layers):
            self.linear_layers.append(nn.Linear(ff_previous_size, configs.d_fc))            
            ff_previous_size = configs.d_fc  
            if self.layer_norm :
                self.layer_norm_module_list.append(nn.LayerNorm(configs.d_fc))        
        
        ## Step 3: final output layer
        self.output_layer = nn.Linear(configs.d_fc, self.pred_len)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        
        # x shape: (batch_size, input_len, n_features)
        batch_size = x_enc.shape[0]
        if x_enc.shape[2] % 2 != 0:
            # Deleting the first coordinate if the dimension is not par, as the action progress is on the first coordinate
            x_enc = x_enc[:,:,1:]  
        x_enc = x_enc.reshape(x_enc.shape[0], x_enc.shape[1], int(x_enc.shape[2] / 2), 2)

        # Step 1: Encoder
        x_enc = self.encoder(x_enc)  # Shape: (batch_size, transformer_embed_dim)

        # Step 2: Fully connected layers       
        for i,linear in enumerate(self.linear_layers):
            residuals = x_enc  
            x_enc = linear(x_enc) 
            if residuals.shape[-1] == x_enc.shape[-1] and self.skip_connections: # Only adding residuals if dimensions match
                x_enc = x_enc + residuals
            if self.layer_norm :
                x_enc = self.layer_norm_module_list[i](x_enc)
            x_enc = self.dropout(x_enc)  
            x_enc = F.relu(x_enc)    
        x_enc = self.output_layer(x_enc)
        x_enc = x_enc.reshape(x_enc.shape[0], x_enc.shape[1], x_enc.shape[2] * x_enc.shape[3])
        x_enc = x_enc[:,-self.pred_len:,:self.c_out]  # Shape: (batch_size, output_len * n_output_feature)   
         
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