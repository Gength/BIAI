import torch
import torch.nn as nn
import torch.nn.functional as F
class MPNNLayer(nn.Module):
    """Single layer of Message Passing Neural Network (MPNN)
    
    Implements message passing and node update operations using:
    - Message function: MLP with ReLU activation
    - Update function: GRUCell for state transition
    
    Args:
        in_dim (int): Dimension of input node features (default: 128)
        hidden_dim (int): Dimension of hidden states (default: 128)
    """
    def __init__(self, in_dim=128, hidden_dim=128):
        super().__init__()
        # Message function: MLP that transforms neighbor features
        self.message_func = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )
        # Update function: GRU cell for node state transition
        self.update_func = nn.GRUCell(hidden_dim, in_dim)

    def forward(self, h, adj):
        """
        Perform one step of message passing and node update
        
        Inputs:
            h: Node features 
                shape [batch_size, num_nodes, in_dim]
            adj: Adjacency matrix (binary or weighted)
                shape [batch_size, num_nodes, num_nodes]
                
        Output: 
            Updated node features
                shape [batch_size, num_nodes, in_dim]
        """
        # Message aggregation: m_v = Σ_{w∈N(v)} MLP(h_w)
        # matmul(adj, message_func(h)) performs neighborhood sum
        m = torch.matmul(adj, self.message_func(h))  # [batch, node, hidden]
        
        # Reshape for GRUCell: flatten batch and node dimensions
        batch_size, num_nodes, in_dim = h.shape
        h_flat = h.reshape(-1, in_dim)              # [batch*node, in_dim]
        m_flat = m.reshape(-1, m.size(-1))           # [batch*node, hidden]
        
        # Update node states: h_v^{t+1} = GRU(h_v^t, m_v)
        updated_flat = self.update_func(m_flat, h_flat)
        
        # Restore original shape
        return updated_flat.view(batch_size, num_nodes, in_dim)


class MPNN(nn.Module):
    """Full MPNN model with multiple message passing steps
    
    Implements:
    - T-step message passing
    - Node-level readout with initial & final features
    - Graph embedding via node summation
    
    Args:
        in_dim (int): Input node feature dimension (default: 128)
        hidden_dim (int): Hidden state dimension (default: 128)
        n_steps (int): Number of message passing steps (default: 5)
        readout_dim (int): Output graph embedding dimension (default: 64)
    """
    def __init__(self, in_dim=128, hidden_dim=128, n_steps=5, readout_dim=64):
        super().__init__()
        self.n_steps = n_steps
        self.mpnn_layer = MPNNLayer(in_dim, hidden_dim)
        
        # Node-level MLP for readout: processes [h_v^0, h_v^T] per node
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, readout_dim),
            nn.ReLU()
        )

    def forward(self, h0, adj):
        """
        Generate graph embedding from initial node features and adjacency
        
        Inputs:
            h0: Initial node features 
                shape [batch_size, num_nodes, in_dim]
            adj: Adjacency matrix 
                shape [batch_size, num_nodes, num_nodes]
                
        Output: 
            Graph embedding vector
                shape [batch_size, readout_dim]
        """
        h = h0
        # Run T steps of message passing
        for _ in range(self.n_steps):
            h = self.mpnn_layer(h, adj)
        
        # Node-level feature fusion: concatenate initial and final states
        # combined_per_node shape: [batch_size, num_nodes, in_dim*2]
        combined_per_node = torch.cat([h0, h], dim=-1)
        
        # Transform each node's combined features
        # node_embeddings shape: [batch_size, num_nodes, readout_dim]
        node_embeddings = self.node_mlp(combined_per_node)
        
        # Generate graph embedding: sum over all nodes
        # graph_embedding shape: [batch_size, readout_dim]
        graph_embedding = torch.sum(node_embeddings, dim=1)
        
        return graph_embedding


class ResNetBlock(nn.Module):
    """ResNet block with 3 convolutional layers per block"""
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
        # First convolution
        out = F.relu(self.bn1(self.conv1(x)))
        # Second convolution
        out = F.relu(self.bn2(self.conv2(out)))
        # Third convolution
        out = self.bn3(self.conv3(out))
        # Residual connection
        out += residual
        return F.relu(out)

class OrderCNN(nn.Module):
    def __init__(self, in_channels=1, num_blocks=3, out_features=32):
        super().__init__()
        # Initial convolutional layer
        self.conv_in = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        
        # Residual blocks (3 blocks)
        self.res_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.res_blocks.add_module(f"res_block_{i}", ResNetBlock(32))
        
        # Additional convolutional layers to reach 11 layers total
        # 1 (initial) + 3 blocks * 3 layers each = 10 layers
        # Add one more convolution to make 11 layers
        self.conv_out = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn_out = nn.BatchNorm2d(32)
        
        # Global max pooling
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Output layer
        self.fc_out = nn.Linear(32, out_features)

    def forward(self, adj):
        # Add channel dimension [batch_size, 1, num_nodes, num_nodes]
        x = adj.unsqueeze(1)
        
        # Initial convolution
        x = F.relu(self.conv_in(x))
        
        # Residual blocks (3 blocks, each with 3 conv layers)
        x = self.res_blocks(x)
        
        # Additional convolution to reach 11 layers
        x = F.relu(self.bn_out(self.conv_out(x)))
        
        # Global pooling [batch_size, 32, 1, 1]
        x = self.pool(x)
        
        # Flatten [batch_size, 32]
        x = x.view(x.size(0), -1)
        
        # Output layer [batch_size, out_features]
        return self.fc_out(x)


class CFGFusionModel(nn.Module):
    """
    Complete semantic-aware model integrating three components:
    1. Semantic-aware modeling (BERT)
    2. Structure-aware modeling (MPNN)
    3. Order-aware modeling (OrderCNN)
    """
    def __init__(self, bert_model, d_model=128, mpnn_readout_dim=64, cnn_out=32, hidden_dim=64, device="cuda"):
        """
        :param bert_model: Pretrained BERT model
        :param d_model: Output embedding dimension of BERT
        :param mpnn_readout_dim: Output dimension of MPNN readout
        :param cnn_out: Output dimension of OrderCNN
        :param hidden_dim: Final graph embedding dimension
        :param device: Computing device
        """
        super().__init__()
        self.device = device
        self.bert = bert_model
        
        # Structure-aware modeling component
        self.mpnn = MPNN(in_dim=d_model, hidden_dim=d_model, n_steps=5, readout_dim=mpnn_readout_dim)
        
        # Order-aware modeling component
        self.order_cnn = OrderCNN(in_channels=1, num_blocks=3, out_features=cnn_out)
        
        # Fusion layer
        fusion_in = mpnn_readout_dim + cnn_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, input_ids, adj_matrix):
        """
        Input:
            input_ids: token sequences [batch_size, num_nodes, seq_len]
            adj_matrix: adjacency matrix [batch_size, num_nodes, num_nodes]
        
        Output:
            graph embedding [batch_size, hidden_dim]
        """
        batch_size, num_nodes, seq_len = input_ids.shape
        
        # === Semantic-aware modeling ===
        # Flatten node dimension [batch_size * num_nodes, seq_len]
        flat_input_ids = input_ids.reshape(batch_size * num_nodes, seq_len)
        
        # Get block embeddings [batch_size * num_nodes, d_model]
        block_embeddings = self.bert.encode(flat_input_ids)
        
        # Restore node dimension [batch_size, num_nodes, d_model]
        node_features = block_embeddings.reshape(batch_size, num_nodes, -1)
        
        # === Structure-aware modeling ===
        # Generate graph structure embedding [batch_size, 64]
        structure_embedding = self.mpnn(node_features, adj_matrix)
        
        # === Order-aware modeling ===
        # Generate order embedding [batch_size, 32]
        order_embedding = self.order_cnn(adj_matrix)
        
        # === Fusion ===
        # Concatenate structure and order embeddings [batch_size, 96]
        combined = torch.cat([structure_embedding, order_embedding], dim=-1)
        
        # Generate final graph embedding [batch_size, hidden_dim]
        graph_embedding = self.fusion(combined)
        
        return graph_embedding
