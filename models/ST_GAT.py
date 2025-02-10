import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.hidden_dim = args.d_model
        self.output_dim = args.c_out
        self.num_agents = args.n_agents
        self.input_dim = args.enc_in // args.n_agents
        self.pred_length = args.pred_len
        self.rec_layers = args.lstm_layers
        self.dropout = args.dropout
        
        # Batch Normalization layers
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim)
            for _ in range(args.e_layers)
        ])
        
        # GAT layers with skip connections
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=self.input_dim if i == 0 else self.hidden_dim,
                out_channels=self.hidden_dim,
                heads=args.n_heads,
                concat=False,
                dropout=args.dropout  # Add dropout to GAT layers
            ) for i in range(args.e_layers)
        ])
        
        # Layer normalization for GRU input
        self.pre_gru_norm = nn.LayerNorm(self.hidden_dim * self.num_agents)
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=self.hidden_dim * self.num_agents,
            hidden_size=self.hidden_dim,
            num_layers=self.rec_layers,
            batch_first=True,
            dropout=args.dropout
        )
        
        # Decoder with improved architecture
        decoder_layers = []
        input_size = args.d_model
        
        for i in range(args.d_layers - 1):
            decoder_layers.extend([
                nn.Linear(input_size, args.d_ff),
                nn.LayerNorm(args.d_ff),  # Add layer normalization
                nn.ReLU(),
                nn.Dropout(args.dropout)
            ])
            input_size = args.d_ff
            
        decoder_layers.extend([
            nn.Linear(input_size, self.pred_length * self.output_dim),
            nn.Dropout(args.dropout)  # Final dropout
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def compute_adjacency(self, x):
        """
        Enhanced adjacency computation with spatial decay
        """
        positions = x[:, :2]
        dist = torch.cdist(positions, positions)
        
        # Gaussian kernel for smooth distance decay
        sigma = dist.mean()  # adaptive sigma
        adj = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
        
        # Sparsify the graph by keeping only top-k connections per node
        k = min(self.num_agents - 1, 8)  # connect to at most k nearest neighbors
        topk_values, _ = torch.topk(adj, k=k, dim=-1)
        threshold = topk_values[..., -1:]
        adj = (adj >= threshold).float()
        
        # Convert to edge_index format
        edge_index = adj.nonzero().t()
        
        return edge_index
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = x_enc[:,:,-self.num_agents * self.input_dim:]
        batch_size = x_enc.shape[0]
        seq_len = x_enc.shape[1]
        
        # Reshape input
        x = x_enc.view(batch_size, seq_len, self.num_agents, -1)
        
        # Process with GAT layers
        gat_outputs = []
        for b in range(batch_size):
            timestep_outputs = []
            for t in range(seq_len):
                x_t = x[b, t]  # (A, D)
                
                edge_index = self.compute_adjacency(x_t)
                
                # GAT processing with skip connections
                x_out = x_t
                for i, (gat_layer, norm_layer) in enumerate(zip(self.gat_layers, self.gat_norms)):
                    identity = x_out
                    
                    # GAT + normalization + activation
                    x_out = gat_layer(x_out, edge_index)
                    x_out = norm_layer(x_out)
                    x_out = F.relu(x_out)
                    
                    # Add skip connection if shapes match
                    if i > 0 and x_out.shape == identity.shape:
                        x_out = x_out + identity
                    
                    # Apply dropout
                    x_out = F.dropout(x_out, p=self.dropout, training=self.training)
                
                timestep_outputs.append(x_out)
            
            timestep_outputs = torch.stack(timestep_outputs)
            gat_outputs.append(timestep_outputs)
        
        gat_outputs = torch.stack(gat_outputs)
        
        # Reshape and normalize for GRU
        gat_outputs = gat_outputs.reshape(batch_size, seq_len, -1)
        gat_outputs = self.pre_gru_norm(gat_outputs)
        
        # Apply GRU
        gru_out, _ = self.gru(gat_outputs)
        last_hidden = gru_out[:, -1]
        
        # Generate predictions
        output = self.decoder(last_hidden)
        output = output.view(batch_size, self.pred_length, self.output_dim)
        
        return output