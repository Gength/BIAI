import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from scipy.sparse import coo_matrix
import numpy as np

class SparseMPNNLayer(MessagePassing):
    """Sparse MPNN layer using PyTorch Geometric"""
    def __init__(self, in_dim=128, hidden_dim=128):
        super().__init__(aggr='add')
        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )
        self.update_gru = nn.GRUCell(hidden_dim, in_dim)
        
    def forward(self, x, edge_index):
        m = self.propagate(edge_index, x=x)
        return self.update_gru(m, x)
    
    def message(self, x_j):
        return self.message_mlp(x_j)


class SparseMPNN(nn.Module):
    """Sparse MPNN model for variable-sized graphs"""
    def __init__(self, in_dim=128, hidden_dim=128, n_steps=5, readout_dim=64):
        super().__init__()
        self.n_steps = n_steps
        self.mpnn_layer = SparseMPNNLayer(in_dim, hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, readout_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, batch):
        h0 = x
        h = x
        for _ in range(self.n_steps):
            h = self.mpnn_layer(h, edge_index)
        combined = torch.cat([h0, h], dim=-1)
        node_embeddings = self.node_mlp(combined)
        return scatter_add(node_embeddings, batch, dim=0)


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out += residual
        return F.relu(out)


class OrderCNN(nn.Module):
    """Order-aware CNN with direct coo_matrix to dense conversion"""
    def __init__(self, in_channels=1, num_blocks=3, out_features=32):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)

        self.res_blocks = nn.Sequential(*[
            ResNetBlock(32) for _ in range(num_blocks)
        ])
        self.conv_out = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn_out = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_out = nn.Linear(32, out_features)

    def forward(self, adj_list):
        embeddings = []
        device = next(self.parameters()).device
        for adj in adj_list:
            adj_dense = torch.tensor(adj.toarray(), dtype=torch.float32, device=device)
            
            # Add necessary dimensions [1, 1, num_nodes, num_nodes]
            x = adj_dense.unsqueeze(0).unsqueeze(0)
            
            # Process through CNN
            x = F.relu(self.conv_in(x))
            x = self.res_blocks(x)
            x = F.relu(self.bn_out(self.conv_out(x)))
            x = self.pool(x)
            x = x.view(1, -1)
            embeddings.append(self.fc_out(x))
            
        return torch.cat(embeddings, dim=0)


class CFGFusionModel(nn.Module):
    """
    Optimized model for batch processing with:
    1. Direct concatenation of input_ids
    2. Efficient coo_matrix handling in OrderCNN
    """
    def __init__(self, bert_model, d_model=128, mpnn_readout_dim=64, 
                 cnn_out=32, hidden_dim=64, device="cuda"):
        super().__init__()
        self.device = device
        self.bert = bert_model
        self.mpnn = SparseMPNN(
            in_dim=d_model, 
            hidden_dim=d_model,
            n_steps=5,
            readout_dim=mpnn_readout_dim
        )
        self.order_cnn = OrderCNN(out_features=cnn_out)
        self.fusion = nn.Sequential(
            nn.Linear(mpnn_readout_dim + cnn_out, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, input_ids_list, adj_list):
        """
        Process a batch of functions in one pass
        
        Args:
            input_ids_list: List of tensors [batch_size], 
                            each [num_nodes, seq_len]
            adj_list: List of coo_matrix [batch_size], 
                      each [num_nodes, num_nodes]
            
        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        # === 1. Semantic-aware modeling ===
        # Directly concatenate all node sequences
        all_nodes = torch.cat(input_ids_list, dim=0).to(self.device)
        block_embeddings = self.bert.encode(all_nodes)  # [total_nodes, d_model]
        
        # === 2. Structure-aware modeling ===
        # Prepare batch information
        batch_vector = []
        edge_indices = []
        node_counts = [nodes.size(0) for nodes in input_ids_list]
        start_idx = 0
        
        for i, (num_nodes, adj) in enumerate(zip(node_counts, adj_list)):
            # Create batch vector
            batch_vector.append(torch.full((num_nodes,), i, device=self.device))
            
            # Build edge_index and apply node offset
            edge_index = torch.tensor(
                np.array([adj.row + start_idx, adj.col + start_idx]),
                dtype=torch.long,
                device=self.device
            )
            edge_indices.append(edge_index)
            start_idx += num_nodes
        
        # Create global tensors
        batch_vector = torch.cat(batch_vector)
        edge_index = torch.cat(edge_indices, dim=1)
        
        # Run sparse MPNN
        structure_embedding = self.mpnn(
            x=block_embeddings,
            edge_index=edge_index,
            batch=batch_vector
        )
        
        # === 3. Order-aware modeling ===
        # Directly process the list of coo_matrix
        order_embedding = self.order_cnn(adj_list)
        
        # === 4. Fusion ===
        combined = torch.cat([structure_embedding, order_embedding], dim=-1)
        return self.fusion(combined)