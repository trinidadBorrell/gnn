"""
MODEL ARCHITECTURE
==================
Purpose: Define PyTorch Geometric GNN model architecture.

Pipeline Position: Used by train.py and inference.py
- Input: None (defines structure only)
- Output: Model class definition

Key Operations:
1. Define graph neural network layers (SAGEConv)
2. Model initialization methods
3. Forward pass logic

This file only defines the model structure, doesn't handle data or training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()

        # Encoder layers
        self.encoder_conv1 = SAGEConv(in_channels, hidden_channels, aggr='add', project = True)
        self.encoder_conv2 = SAGEConv(hidden_channels, latent_dim, aggr='add', project = True)
        
        # Decoder layers
        self.decoder_conv1 = SAGEConv(latent_dim, hidden_channels, aggr='add',project = True)
        self.decoder_conv2 = SAGEConv(hidden_channels, in_channels, aggr='add', project = True)

    def encode(self, x, edge_index):
        x = self.encoder_conv1(x, edge_index)
        x = F.relu(x)
        x = self.encoder_conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        x = self.decoder_conv1(z, edge_index)
        x = F.relu(x)
        x = self.decoder_conv2(x, edge_index)
        return x
       
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_reconstructed = self.decode(z, edge_index)
        return x_reconstructed, z 