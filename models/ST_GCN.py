import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN




class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.seq_len = config.seq_len
        self.hidden_dim = config.d_model
        self.output_dim = config.c_out  # 1
        self.num_agents = config.n_agents  # A
        self.input_dim = config.enc_in // config.n_agents  # D
        self.pred_length = config.pred_len
        self.rec_layers = config.lstm_layers        
        self.d_ff = config.d_ff
        
        self.recurrent = TGCN(self.input_dim, self.hidden_dim)
        self.avg_pool1 = torch.nn.AvgPool1d(self.num_agents, stride=1)
        self.avg_pool2 = torch.nn.AvgPool1d(self.seq_len, stride=1)
        self.dropout = torch.nn.Dropout()
        
        # Linear Decoder
        decoder_layers = []
        input_size = config.d_model
        # Create multiple linear layers if specified
        for i in range(config.d_layers - 1):
            decoder_layers.extend([
                nn.Linear(config.d_model, config.d_ff),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_size = config.d_ff
            
        # Final output layer
        decoder_layers.append(nn.Linear(input_size, self.pred_length * self.output_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def compute_adjacency(self, x):
        """
        Compute adjacency matrix with inverse distance weighting.
        
        Args:
            x (torch.Tensor): Input tensor of shape [A, D] for one timestep
            
        Returns:
            edge_index (torch.Tensor): COO format edge indices [2, num_edges]
            edge_weight (torch.Tensor): Edge weights [num_edges]
        """
        # Compute pairwise distances between agents
        # Using all dimensions for distance computation
        distances = torch.cdist(x, x)
        
        # Create inverse distance weights (adding small epsilon to avoid division by zero)
        weights = 1.0 / (distances + 1e-6)
        
        # Set diagonal to 0 to remove self-loops if needed
        # weights.fill_diagonal_(0)
        
        # Convert to edge_index (COO format) and edge_weight
        edges = torch.nonzero(torch.ones_like(weights), as_tuple=True)
        edge_index = torch.stack(edges)
        edge_weight = weights[edges]
        
        return edge_index, edge_weight

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, A*D]
            
        Returns:
            y_out (torch.Tensor): Output predictions
        """
        batch_size, seq_len, total_features = x.shape
        # Reshape to separate agents: [B, L, A, D]
        x_reshaped = x.view(batch_size, seq_len, self.num_agents, self.input_dim)
        
        out = []
        # Process each batch
        for b in range(batch_size):
            batch_out = []
            # Process each timestep
            for t in range(seq_len):
                # Get current timestep data [A, D]
                current_x = x_reshaped[b, t]
                
                # Compute adjacency for current timestep
                edge_index, edge_weight = self.compute_adjacency(current_x)
                
                # Apply TGCN
                h = self.recurrent(current_x, edge_index, edge_weight)
                h = F.relu(h)
                h = (h.t()).unsqueeze(1)
                h = torch.flatten(h)
                batch_out.append(h)
            
            # Stack timesteps
            batch_out = torch.stack(batch_out)
            out.append(batch_out)
        
        # Stack batches
        out = torch.stack(out)
        
        # Further processing
        out_s = torch.squeeze(out.transpose(0, 1)).unsqueeze(1)
        y_out = self.avg_pool2(out_s)
        y_out = self.decoder(y_out.squeeze())
        
        return y_out.view(batch_size, self.pred_length, self.output_dim)
