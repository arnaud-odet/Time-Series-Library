import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding

class Model(nn.Module):
    """
    Alternative implementation of 2D Transformer for time series forecasting
    
    This version uses a more direct 2D approach by:
    1. Applying attention across features for each time step
    2. Then applying attention across time steps for the processed features
    
    This avoids the loop in the previous implementation for better efficiency.
    """
    def __init__(self, configs):
        self.task_name = configs.task_name
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        
        # Embedding layers
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                          configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                          configs.dropout)
        
        # Feature transformer layers
        self.feature_attn_layers = nn.ModuleList([
            AttentionLayer(
                FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                configs.d_model, configs.n_heads)
            for _ in range(configs.e_layers // 2)
        ])
        
        self.feature_ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU() if configs.activation == 'gelu' else nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_ff, configs.d_model),
                nn.Dropout(configs.dropout)
            )
            for _ in range(configs.e_layers // 2)
        ])
        
        # Temporal transformer layers
        self.temporal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                     output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers // 2)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Layer norms
        self.feature_norms = nn.ModuleList([
            nn.LayerNorm(configs.d_model)
            for _ in range(configs.e_layers // 2)
        ])
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                     output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                     output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Output projection
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecast function with more efficient 2D transformer implementation
        """
        # Input shape: [B, T, D]
        B, T, _ = x_enc.shape
        
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, d_model]
        
        # Process features at each time step (feature dimension attention)
        # We transpose to [B, d_model, T] and process each row (feature) independently
        feature_out = enc_out
        
        # Apply feature attention layers
        for i, (attn_layer, ff_layer, norm_layer) in enumerate(zip(
            self.feature_attn_layers, self.feature_ff_layers, self.feature_norms)):
            
            # Apply feature attention to each time step separately
            for t in range(T):
                # Get features at time t for all batches: [B, d_model]
                x_t = feature_out[:, t, :].unsqueeze(1)  # [B, 1, d_model]
                
                # Self-attention on feature dimension
                attn_out, _ = attn_layer(x_t, x_t, x_t, attn_mask = None)  # [B, 1, d_model] # HERE
                
                # Add & Norm
                x_t = x_t + attn_out  # [B, 1, d_model]
                x_t = norm_layer(x_t)  # [B, 1, d_model]
                
                # Feed Forward
                ff_out = ff_layer(x_t)  # [B, 1, d_model]
                
                # Add & Norm
                x_t = x_t + ff_out  # [B, 1, d_model]
                x_t = norm_layer(x_t)  # [B, 1, d_model]
                
                # Update the time step
                feature_out[:, t, :] = x_t.squeeze(1)  # [B, d_model]
        
        # Now apply temporal attention to capture dependencies across time
        temporal_out, _ = self.temporal_encoder(feature_out)  # [B, T, d_model]
        
        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [B, T', d_model]
        dec_out = self.decoder(dec_out, temporal_out)  # [B, T', d_model]
        
        # Final projection
        dec_out = self.projection(dec_out)  # [B, T', C_out]
        
        return dec_out
