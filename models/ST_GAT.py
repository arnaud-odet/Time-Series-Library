import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class Model(nn.Module):
    def __init__(self, 
                 args):
        super(Model, self).__init__()
        
        self.hidden_dim = args.d_model
        self.output_dim = args.c_out  # 1
        self.num_agents = args.n_agents  # A
        self.input_dim = args.enc_in // args.n_agents # D
        self.pred_length = args.pred_len
        self.rec_layers = args.lstm_layers
        # P
        
        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=self.input_dim if i == 0 else self.hidden_dim,
                out_channels=self.hidden_dim,
                heads=args.n_heads,
                concat=False
            ) for i in range(args.e_layers)
        ])
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=self.hidden_dim * self.num_agents,
            hidden_size=self.hidden_dim,
            num_layers=self.rec_layers,
            batch_first=True,
            dropout = args.dropout
        )
        
        # Linear Decoder
        decoder_layers = []
        input_size = args.d_model
        # Create multiple linear layers if specified
        for i in range(args.d_layers - 1):
            decoder_layers.extend([
                nn.Linear(args.d_model, args.d_ff),
                nn.ReLU(),
                nn.Dropout(args.dropout)
            ])
            input_size = args.d_ff
            
        # Final output layer
        decoder_layers.append(nn.Linear(input_size, self.pred_length * self.output_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def compute_adjacency(self, x):
        """
        Compute adjacency matrix based on spatial distances
        x shape: (N, D) where N = num_agents
        Returns edge_index for GAT
        """
        # Extract x,y coordinates
        positions = x[:, :2]  # (N, 2)
        
        # Compute pairwise distances
        dist = torch.cdist(positions, positions)  # (N, N)
        
        # Create edge_index based on distances (connect if distance < threshold)
        threshold = dist.mean() + dist.std()  # adaptive threshold
        adj = (dist < threshold).float()
        
        # Convert to edge_index format
        edge_index = adj.nonzero().t()  # (2, num_edges)
        
        return edge_index
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x shape: (B, L, A*D)+
        """
        x_enc = x_enc[:,:,1:]
        batch_size = x_enc.shape[0]
        seq_len = x_enc.shape[1]
        
        # Reshape input to (B*L, A, D)
        x = x_enc.view(batch_size, seq_len, self.num_agents, -1)
        
        # Process each batch and timestep with GAT
        gat_outputs = []
        for b in range(batch_size):
            timestep_outputs = []
            for t in range(seq_len):
                # Get current timestep data
                x_t = x[b, t]  # (A, D)
                
                # Compute edge_index for current timestep
                edge_index = self.compute_adjacency(x_t)
                
                # Process through GAT layers
                x_out = x_t  # (A, D)
                for gat_layer in self.gat_layers:
                    x_out = gat_layer(x_out, edge_index)
                    x_out = F.relu(x_out)
                
                timestep_outputs.append(x_out)
            
            # Stack timesteps
            timestep_outputs = torch.stack(timestep_outputs)  # (L, A, hidden_dim)
            gat_outputs.append(timestep_outputs)
        
        # Stack batches
        gat_outputs = torch.stack(gat_outputs)  # (B, L, A, hidden_dim)
        
        # Reshape for GRU
        gat_outputs = gat_outputs.reshape(batch_size, seq_len, -1)  # (B, L, A*hidden_dim)
        
        # Apply GRU
        gru_out, _ = self.gru(gat_outputs)  # (B, L, hidden_dim)
        
        # Take the last hidden state
        last_hidden = gru_out[:, -1]  # (B, hidden_dim)
        
        # Generate predictions
        output = self.decoder(last_hidden)  # (B, P*output_dim)
        output = output.view(batch_size, self.pred_length, self.output_dim)
        
        return output
