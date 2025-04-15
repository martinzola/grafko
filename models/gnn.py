import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool, global_add_pool, global_max_pool
from typing import Dict, Any, Optional, List


class GNNBlock(torch.nn.Module):
    """Basic GNN block with configurable layer type."""
    
    def __init__(self, in_channels: int, out_channels: int, layer_type: str = 'GCN', 
                 dropout_rate: float = 0.1, activation: str = 'relu'):
        super(GNNBlock, self).__init__()
        
        # Layer type selection
        if layer_type == 'GCN':
            self.conv = GCNConv(in_channels, out_channels)
        elif layer_type == 'GAT':
            self.conv = GATConv(in_channels, out_channels)
        elif layer_type == 'GraphConv':
            self.conv = GraphConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
            
        # Activation selection
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
    def forward(self, x, edge_index, edge_attr=None):
        # Apply convolution
        if isinstance(self.conv, GATConv):
            x = self.conv(x, edge_index)
        else:
            x = self.conv(x, edge_index, edge_attr)
            
        # Apply batch norm, activation and dropout
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class MolecularGNN(torch.nn.Module):
    """Molecular GNN for classification or regression tasks."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 1,
                 num_layers: int = 3, dropout_rate: float = 0.1,
                 layer_type: str = 'GCN', activation: str = 'relu',
                 pooling: str = 'mean', task_type: str = 'classification'):
        """
        Args:
            in_channels: Input feature dimensions
            hidden_channels: Hidden layer dimensions
            out_channels: Output dimensions (1 for binary classification)
            num_layers: Number of GNN layers
            dropout_rate: Dropout rate
            layer_type: Type of GNN layer ('GCN', 'GAT', 'GraphConv')
            activation: Activation function ('relu', 'leaky_relu', 'elu')
            pooling: Graph pooling strategy ('mean', 'sum', 'max')
            task_type: Task type ('classification' or 'regression')
        """
        super(MolecularGNN, self).__init__()
        
        self.task_type = task_type
        
        # Input embedding layer (to transform discrete features to continuous)
        self.embedding = nn.Linear(in_channels, hidden_channels)
        
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GNNBlock(hidden_channels, hidden_channels, layer_type, dropout_rate, activation))
        
        for _ in range(num_layers - 1):
            self.convs.append(GNNBlock(hidden_channels, hidden_channels, layer_type, dropout_rate, activation))
        
        # Pooling function
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, x, edge_index, batch, edge_attr=None):
        """Forward pass through the network.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            edge_attr: Edge features [num_edges, edge_attr_dim]
            
        Returns:
            Predicted molecule properties
        """
        # Initial embedding for atom types
        x = self.embedding(x.float())  # Convert discrete features to float
        
        # Apply GNN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        
        # Pooling
        x = self.pool(x, batch)
        
        # MLP head
        x = self.mlp(x)
        
        return x


class AtomEmbedding(torch.nn.Module):
    """Atom embedding layer for discrete atom features."""
    
    def __init__(self, atom_types: List[int], embedding_dim: int):
        super(AtomEmbedding, self).__init__()
        self.atom_embedding = nn.Embedding(max(atom_types) + 1, embedding_dim)
        self.fc = nn.Linear(embedding_dim + 3, embedding_dim)  # 3 for other features
        
    def forward(self, x):
        # x shape: [num_nodes, 4] where x[:,0] is atomic_number
        atom_type = x[:, 0].long()
        atom_embedding = self.atom_embedding(atom_type)
        
        # Concatenate with other features
        other_features = x[:, 1:].float()  # Formal charge, hybridization, aromaticity
        combined = torch.cat([atom_embedding, other_features], dim=1)
        
        # Project to embedding dimension
        return self.fc(combined)


class MoleculeEncoder(torch.nn.Module):
    """Encoder for molecular graphs with atom embeddings."""
    
    def __init__(self, atom_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, dropout_rate: float = 0.1,
                 layer_type: str = 'GCN', pooling: str = 'mean'):
        super(MoleculeEncoder, self).__init__()
        
        # Atom embedding
        self.atom_embedding = nn.Linear(atom_dim, hidden_dim)
        
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GNNBlock(hidden_dim, hidden_dim, layer_type, dropout_rate))
        
        for _ in range(num_layers - 1):
            self.convs.append(GNNBlock(hidden_dim, hidden_dim, layer_type, dropout_rate))
        
        # Pooling function
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
            
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, batch, edge_attr=None):
        # Embed atoms
        x = self.atom_embedding(x.float())
        
        # Apply GNN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Project to output dimension
        return self.output_proj(x)


class MolecularGNNFactory:
    """Factory for creating molecular GNN models."""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> torch.nn.Module:
        """Create a molecular GNN model based on type and config.
        
        Args:
            model_type: Type of model ('basic_gnn', 'encoder', etc.)
            config: Model configuration dictionary
            
        Returns:
            Instantiated model
        """
        if model_type == 'basic_gnn':
            return MolecularGNN(
                in_channels=config.get('in_channels', 4),
                hidden_channels=config.get('hidden_channels', 64),
                out_channels=config.get('out_channels', 1),
                num_layers=config.get('num_layers', 3),
                dropout_rate=config.get('dropout_rate', 0.1),
                layer_type=config.get('layer_type', 'GCN'),
                activation=config.get('activation', 'relu'),
                pooling=config.get('pooling', 'mean'),
                task_type=config.get('task_type', 'classification')
            )
        elif model_type == 'encoder':
            return MoleculeEncoder(
                atom_dim=config.get('atom_dim', 4),
                hidden_dim=config.get('hidden_dim', 64),
                output_dim=config.get('output_dim', 32),
                num_layers=config.get('num_layers', 3),
                dropout_rate=config.get('dropout_rate', 0.1),
                layer_type=config.get('layer_type', 'GCN'),
                pooling=config.get('pooling', 'mean')
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")