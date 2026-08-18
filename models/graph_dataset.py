"""Optimised variable-size graph datasets used by Task 1 and Task 2.

The pretraining dataset stays in :mod:`models.dataset`.  Keeping the graph-only
fast path separate also lets pipeline.py distinguish pretraining dependencies
from downstream fine-tuning dependencies.
"""
import torch

from models.dataset import (
    FunctionPairDataset as _FunctionPairDataset,
    Task2Dataset as _Task2Dataset,
)


def _encode_blocks(tokenizer, blocks, seq_len):
    """Encode an entire CFG without the per-block HF dispatch overhead."""
    input_ids = torch.full(
        (len(blocks), seq_len), tokenizer.pad_token_id, dtype=torch.long)
    content_length = max(seq_len - 2, 0)
    for row, block in enumerate(blocks):
        tokens = tokenizer._tokenize(block)[:content_length]
        ids = [tokenizer.cls_token_id]
        ids.extend(tokenizer.vocab.get(token, tokenizer.unk_token_id)
                   for token in tokens)
        if seq_len > 1:
            ids.append(tokenizer.sep_token_id)
        ids = ids[:seq_len]
        input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return input_ids


def _process_graph(dataset, func_data):
    """Encode all blocks and retain the adjacency as sparse COO."""
    blocks = func_data["instruction_blocks"]
    if not blocks:
        raise ValueError("a CFG must contain at least one basic block")
    input_ids = _encode_blocks(dataset.tokenizer, blocks, dataset.seq_len)

    adj_data = func_data["adjacency_matrix"]
    indices = torch.tensor(
        [adj_data["row"], adj_data["col"]], dtype=torch.long)
    values = torch.tensor(adj_data["data"], dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        indices, values, size=tuple(adj_data["shape"]),
        check_invariants=False).coalesce()
    return input_ids, adjacency


class FunctionPairDataset(_FunctionPairDataset):
    """Task 1 dataset with one preallocated token matrix per CFG."""

    def process_function(self, func_data):
        return _process_graph(self, func_data)

    def graph_sizes(self):
        """Preload compact per-pair work estimates for batch scheduling.

        BERT work is roughly linear in CFG nodes (token length is fixed), while
        OrderCNN also sees an ``N x N`` adjacency.  The quadratic term is
        converted to a node-equivalent scale so neither component completely
        dominates scheduling.  Only integers are retained; token matrices and
        adjacencies continue to be loaded lazily by DataLoader workers.
        """
        if getattr(self, "_graph_sizes", None) is None:
            costs = []
            shapes = []
            for row in self.df.itertuples(index=False):
                a_key = (row.anchor_function_name, row.anchor_compiler,
                         str(row.anchor_version), row.anchor_opt,
                         row.anchor_arch, row.anchor_function_file)
                t_key = (row.target_function_name, row.target_compiler,
                         str(row.target_version), row.target_opt,
                         row.target_arch, row.target_function_file)
                a_n = len(self.dataset[self.mapping[a_key]]["instruction_blocks"])
                t_n = len(self.dataset[self.mapping[t_key]]["instruction_blocks"])
                costs.append(a_n + t_n + (a_n * a_n + t_n * t_n) // 128)
                shapes.append((a_n, t_n))
            self._graph_sizes = costs
            self._graph_shapes = shapes
        return self._graph_sizes

    def graph_shapes(self):
        """Native anchor/target sizes used for exact-shape fused batches."""
        self.graph_sizes()
        return self._graph_shapes


class Task2Dataset(_Task2Dataset):
    """Task 2 dataset with the same direct tokenization/sparse transfer path."""

    def process_function(self, func_data):
        return _process_graph(self, func_data)

    def graph_sizes(self):
        """Compact BERT + OrderCNN work estimates used by the sampler."""
        if getattr(self, "_graph_sizes", None) is None:
            costs = []
            shapes = []
            for key in self.functions:
                n = len(self.dataset[self.mapping[key]]["instruction_blocks"])
                costs.append(n + n * n // 128)
                shapes.append(n)
            self._graph_sizes = costs
            self._graph_shapes = shapes
        return self._graph_sizes

    def graph_shapes(self):
        self.graph_sizes()
        return self._graph_shapes
