import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        
        # LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=configs.dropout if configs.e_layers > 1 else 0
        )
        
        # Linear Decoder
        decoder_layers = []
        input_size = configs.d_model
        
        # Create multiple linear layers if specified
        for i in range(configs.d_layers - 1):
            decoder_layers.extend([
                nn.Linear(input_size, configs.d_ff),
                nn.ReLU(),
                nn.Dropout(configs.dropout)
            ])
            input_size = configs.d_ff
            
        # Final output layer
        decoder_layers.append(nn.Linear(input_size, configs.c_out))
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc shape: [batch_size, seq_len, enc_in]
        
        # LSTM encoding
        enc_out, (hidden, cell) = self.encoder(x_enc)
        # enc_out shape: [batch_size, seq_len, d_model]
        
        # Take the last sequence output for forecasting
        last_hidden = enc_out[:, -1, :]  # [batch_size, d_model]
        
        # Repeat the hidden state for each prediction step
        batch_size = x_enc.size(0)
        decoder_input = last_hidden.unsqueeze(1).repeat(1, self.pred_len, 1)
        # decoder_input shape: [batch_size, pred_len, d_model]
        
        # Reshape for decoder
        decoder_input = decoder_input.reshape(batch_size * self.pred_len, -1)
        
        # Pass through decoder
        output = self.decoder(decoder_input)
        
        # Reshape output
        output = output.reshape(batch_size, self.pred_len, -1)
        
        return output


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]