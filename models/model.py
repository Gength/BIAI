import torch
import torch.nn as nn
import torch.nn.functional as F
class MPNNLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.message_func = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )
        self.update_func = nn.GRUCell(hidden_dim, in_dim)

    def forward(self, h, adj):
        """
        Inputs:
            h: node features [batch_size, num_nodes, in_dim]
            adj: adjacency matrix [batch_size, num_nodes, num_nodes]
        Output: 
            updated node features [batch_size, num_nodes, in_dim]
        """
        # Message aggregation: m_v = Σ_{w∈N(v)} MLP(h_w)
        m = torch.matmul(adj, self.message_func(h))
        # Update node features: h_v = GRU(h_v, m_v)
        batch_size, num_nodes, _ = h.shape
        h_flat = h.reshape(-1, h.size(-1))
        m_flat = m.reshape(-1, m.size(-1))
        updated_flat = self.update_func(m_flat, h_flat)
        return updated_flat.reshape(batch_size, num_nodes, -1)

class MPNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_steps=5):
        super().__init__()
        self.n_steps = n_steps
        self.mpnn_layer = MPNNLayer(in_dim, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU()
        )

    def forward(self, h0, adj):
        """
        Inputs:
            h0: initial node features [batch_size, num_nodes, in_dim]
            adj: adjacency matrix [batch_size, num_nodes, num_nodes]
        Output: 
            graph embedding [batch_size, hidden_dim]
        """
        h = h0
        # Run T steps of message passing
        for _ in range(self.n_steps):
            h = self.mpnn_layer(h, adj)
        
        # Read out features at step 0 and step T and concatenate
        h0_sum = torch.sum(h0, dim=1)  # [batch_size, in_dim]
        hT_sum = torch.sum(h, dim=1)   # [batch_size, in_dim]
        combined = torch.cat([h0_sum, hT_sum], dim=1)
        
        # Generate graph embedding
        return self.readout(combined)

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Input: 
            x: feature maps [batch_size, channels, height, width]
        Output: 
            transformed features [batch_size, channels, height, width]
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class OrderCNN(nn.Module):
    def __init__(self, in_channels=1, num_blocks=3, out_features=32):
        super().__init__()
        # Initial convolutional layer
        self.conv_in = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(32)
        
        # Residual blocks
        self.res_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.res_blocks.add_module(f"res_block_{i}", ResNetBlock(32))
        
        # Global max pooling
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Output layer
        self.fc_out = nn.Linear(32, out_features)

    def forward(self, adj):
        """
        Input: 
            adj: adjacency matrix [batch_size, num_nodes, num_nodes]
        Output: 
            order embedding [batch_size, out_features]
        """
        # Add channel dimension [batch_size, 1, num_nodes, num_nodes]
        x = adj.unsqueeze(1)
        
        # Initial convolution
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        # Residual blocks
        x = self.res_blocks(x)
        
        # Global pooling [batch_size, 32, 1, 1]
        x = self.pool(x)
        
        # Flatten [batch_size, 32]
        x = x.view(x.size(0), -1)
        
        # Output layer [batch_size, out_features]
        return self.fc_out(x)


class SemanticAwareModel(nn.Module):
    """
    Complete semantic-aware model integrating three components:
    1. Semantic-aware modeling (BERT)
    2. Structure-aware modeling (MPNN)
    3. Order-aware modeling (OrderCNN)
    """
    def __init__(self, bert_model, d_model=128, hidden_dim=64, device="cuda"):
        """
        :param bert_model: Pretrained BERT model
        :param d_model: Output embedding dimension of BERT
        :param hidden_dim: Final graph embedding dimension
        :param device: Computing device
        """
        super().__init__()
        self.device = device
        self.bert = bert_model
        
        # Structure-aware modeling component
        self.mpnn = MPNN(in_dim=d_model, hidden_dim=d_model, n_steps=5)
        
        # Order-aware modeling component
        self.order_cnn = OrderCNN(in_channels=1, num_blocks=3)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 32, hidden_dim),  # 32 is the output dimension of OrderCNN
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
        # Generate graph structure embedding [batch_size, d_model]
        structure_embedding = self.mpnn(node_features, adj_matrix)
        
        # === Order-aware modeling ===
        # Generate order embedding [batch_size, 32]
        order_embedding = self.order_cnn(adj_matrix)
        
        # === Fusion ===
        # Concatenate structure and order embeddings [batch_size, d_model + 32]
        combined = torch.cat([structure_embedding, order_embedding], dim=-1)
        
        # Generate final graph embedding [batch_size, hidden_dim]
        graph_embedding = self.fusion(combined)
        
        return graph_embedding

class SiameseNetwork(nn.Module):
    def __init__(self, semantic_model, graph_hidden_dim=64):
        super().__init__()
        self.semantic_model = semantic_model
        self.classifier = nn.Sequential(
            nn.Linear(2 * graph_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid() # binary classification
        )

    def forward(self, a_ids, a_adj, t_ids, t_adj):
        # Get graph embeddings
        a_embed = self.semantic_model(a_ids, a_adj)
        t_embed = self.semantic_model(t_ids, t_adj)
        
        # Concatenate embeddings and classify
        combined = torch.cat([a_embed, t_embed], dim=1)
        return self.classifier(combined).squeeze()