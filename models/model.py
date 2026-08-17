"""Fine-tuning model: semantic-aware (BERT) + structural-aware (MPNN+GRU)
+ order-aware (ResNet on the adjacency matrix), fused into a graph embedding.

This matches the Order-Matters paper:
    g_final = MLP([g_ss, g_o])
with g_ss from an MPNN (GRU update, sum readout over [h^0, h^T]) and g_o
from an 11-layer ResNet with global max pooling on the adjacency matrix.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MPNNLayer(nn.Module):
    """Message function MLP + GRU update (one message passing step)."""

    def __init__(self, in_dim=128, hidden_dim=128):
        super().__init__()
        # m_v^{t+1} = sum_{w in N(v)} MLP(h_w^t)
        self.message_func = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        # h_v^{t+1} = GRU(h_v^t, m_v^{t+1})
        self.update_func = nn.GRUCell(hidden_dim, in_dim)

    def forward(self, h, adj):
        """
        h:   [B, N, in_dim] node features
        adj: [B, N, N] adjacency matrix
        """
        m = torch.matmul(adj, self.message_func(h))  # [B, N, hidden]
        batch_size, num_nodes, in_dim = h.shape
        h_flat = h.reshape(-1, in_dim)
        m_flat = m.reshape(-1, m.size(-1))
        updated = self.update_func(m_flat, h_flat)
        return updated.view(batch_size, num_nodes, in_dim)


class MPNN(nn.Module):
    """T-step message passing with sum readout over [h^0, h^T]."""

    def __init__(self, in_dim=128, hidden_dim=128, n_steps=5, readout_dim=64):
        super().__init__()
        self.n_steps = n_steps
        self.mpnn_layer = MPNNLayer(in_dim, hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, readout_dim),
            nn.ReLU(),
        )

    def forward(self, h0, adj):
        h = h0
        for _ in range(self.n_steps):
            h = self.mpnn_layer(h, adj)
        combined = torch.cat([h0, h], dim=-1)          # [B, N, 2*in]
        node_embeddings = self.node_mlp(combined)      # [B, N, readout]
        return torch.sum(node_embeddings, dim=1)       # [B, readout]


class ResNetBlock(nn.Module):
    """Residual block with 3 conv layers (3x3, padding=1)."""

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
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + residual)


class OrderCNN(nn.Module):
    """11-layer ResNet (3 residual blocks) on the adjacency matrix,
    followed by global max pooling -> order embedding g_o."""

    def __init__(self, in_channels=1, num_blocks=3, out_features=32):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.res_blocks.add_module(f"res_block_{i}", ResNetBlock(32))
        self.conv_out = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn_out = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_out = nn.Linear(32, out_features)

    def forward(self, adj):
        x = adj.unsqueeze(1)                      # [B, 1, N, N]
        x = F.relu(self.conv_in(x))
        x = self.res_blocks(x)
        x = F.relu(self.bn_out(self.conv_out(x)))
        x = self.pool(x)                          # [B, 32, 1, 1]
        x = x.view(x.size(0), -1)
        return self.fc_out(x)                     # [B, out_features]


def _dense_adjacency(adj, device):
    """Convert one native-size adjacency matrix to a dense float tensor."""
    if torch.is_tensor(adj):
        dense = adj.to_dense() if adj.is_sparse else adj
        return dense.to(device=device, dtype=torch.float32)
    return torch.as_tensor(adj.toarray(), dtype=torch.float32, device=device)


class CFGFusionModel(nn.Module):
    """BERT block embeddings + MPNN + OrderCNN fused into a graph embedding.

    Args:
        bert: a `BERTForPretraining` (or any module exposing
            `encode_block_embeddings(input_ids, attention_mask, token_type_ids)`).
        d_model: BERT hidden size (block embedding dim).
        mpnn_readout_dim: MPNN graph embedding dim.
        cnn_out: OrderCNN embedding dim.
        hidden_dim: final graph embedding dim.
    """

    def __init__(self, bert, d_model=128, mpnn_readout_dim=64, cnn_out=32,
                 hidden_dim=64):
        super().__init__()
        self.bert = bert
        self.mpnn = MPNN(in_dim=d_model, hidden_dim=d_model, n_steps=5,
                         readout_dim=mpnn_readout_dim)
        self.order_cnn = OrderCNN(in_channels=1, num_blocks=3, out_features=cnn_out)
        fusion_in = mpnn_readout_dim + cnn_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(),
        )
        # Gradient checkpointing for the BERT encoder: with variable-size
        # graphs, large CFGs (up to ~1000 blocks) encode ~1000 sequences per
        # graph; without checkpointing the attention activations alone exceed
        # 16GB. Checkpointing trades a little compute for a lot of memory.
        try:
            self.bert.bert.gradient_checkpointing_enable()
        except Exception:
            pass

    def forward(self, input_ids, adj_matrix):
        """Graph embeddings for a batch of CFGs.

        Accepts either:
        - batched tensors [B, N, L] + [B, N, N] (all graphs same size), or
        - lists of per-graph tensors with native, possibly different sizes
          (paper: CNN works on variable-size inputs without padding/clipping).
        Returns [B, hidden_dim].
        """
        if isinstance(input_ids, (list, tuple)):
            if len(input_ids) != len(adj_matrix):
                raise ValueError("input_ids and adj_matrix must have equal lengths")
            # The paper explicitly applies the CNN to native, variable-size
            # adjacency matrices without padding or clipping. Process every
            # graph independently, then concatenate graph embeddings. This
            # also guarantees that an embedding cannot depend on which other
            # graph happens to share its DataLoader batch.
            embeddings = []
            for ids, adj in zip(input_ids, adj_matrix):
                if ids.ndim != 2:
                    raise ValueError("each graph input must have shape [N, L]")
                dense_adj = _dense_adjacency(adj, ids.device)
                if dense_adj.shape != (ids.size(0), ids.size(0)):
                    raise ValueError("adjacency shape must match the node count")
                embeddings.append(
                    self._forward_batch(ids.unsqueeze(0), dense_adj.unsqueeze(0))
                )
            if not embeddings:
                raise ValueError("at least one graph is required")
            return torch.cat(embeddings, dim=0)

        if not torch.is_tensor(input_ids) or not torch.is_tensor(adj_matrix):
            raise TypeError("inputs must both be tensors or per-graph sequences")
        return self._forward_batch(input_ids, adj_matrix)

    def _forward_batch(self, input_ids, adj_matrix):
        """
        input_ids:  [B, N, L] token ids of each block (same N across batch)
        adj_matrix: [B, N, N] adjacency matrix
        Returns:    [B, hidden_dim] graph embeddings
        """
        batch_size, num_nodes, _ = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * num_nodes, -1)
        attention_mask = (flat_ids != self.bert.config.pad_token_id).long()
        token_type_ids = torch.zeros_like(flat_ids)

        block_embeddings = self.bert.encode_block_embeddings(
            flat_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # [B*N, d_model]
        node_features = block_embeddings.reshape(batch_size, num_nodes, -1)

        structure_embedding = self.mpnn(node_features, adj_matrix)  # [B, 64]
        order_embedding = self.order_cnn(adj_matrix)                # [B, 32]
        combined = torch.cat([structure_embedding, order_embedding], dim=-1)
        return self.fusion(combined)                                # [B, hidden]

    def save_pretrained(self, save_path):
        """Save the whole model (BERT included) to `save_path`."""
        import os
        os.makedirs(save_path, exist_ok=True)
        self.bert.config.save_pretrained(save_path)
        torch.save(
            {"model": self.state_dict(), "bert_config": self.bert.config.to_dict()},
            os.path.join(save_path, "pytorch_model.bin"),
        )
