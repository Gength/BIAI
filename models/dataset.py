"""Datasets for pre-training (per-function CFG samples) and fine-tuning
(function pair samples). Tokenization goes through the HuggingFace
tokenizer API (see `AsmTokenizer`).
"""
import os
import pickle
import heapq
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from scipy.sparse import coo_matrix

opt = ["O0", "O1", "O2", "O3"]
architectures = ["x64", "arm64"]  # paper: x86-64 & ARM
opt_arch_combinations = [(o, arch) for o in opt for arch in architectures]
opt_arch_mapping = {(o, arch): i for i, (o, arch) in enumerate(opt_arch_combinations)}


class TaskDataset(Dataset):
    """Wrap the dataset and add task type information."""

    def __init__(self, dataset, task_type):
        self.dataset = dataset
        self.task_type = task_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["task_type"] = self.task_type
        return item


class FunctionDataset(Dataset):
    """Per-function CFG samples loaded from the JSONL dataset."""

    def __init__(self, dataset_path, function_idx_mapping_path, tokenizer,
                 max_len=128):
        self.tokenizer = tokenizer
        self.dataset = load_dataset(
            "json",
            data_files=dataset_path,
            split="train",
            cache_dir=os.path.join(".", "outputs", "cache"),
            keep_in_memory=False,
        )
        with open(function_idx_mapping_path, "rb") as f:
            self.function_idx_mapping = pickle.load(f)
        self.idx_function_mapping = {
            v: k for k, v in self.function_idx_mapping.items()
        }
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        key = self.idx_function_mapping[idx]  # (name, compiler, version, opt, arch, file)
        data["opt_arch_idx"] = opt_arch_mapping[(data["opt"], data["arch"])]
        return data


class FunctionPairDataset(Dataset):
    """(anchor, target) function pairs for the siamese fine-tuning task."""

    def __init__(self, function_pool_path, dataset_path, function_idx_mapping_path,
                 tokenizer, seq_len=128, max_nodes=None):
        # max_nodes kept for API compatibility; variable-size graphs are used
        # (no padding/clipping), so it is ignored.
        self.df = pd.read_csv(function_pool_path)
        self.dataset = load_dataset(
            "json",
            data_files=dataset_path,
            split="train",
            cache_dir=os.path.join(".", "outputs", "cache"),
            keep_in_memory=False,
        )
        with open(function_idx_mapping_path, "rb") as f:
            self.mapping = pickle.load(f)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        a_key = (row["anchor_function_name"], row["anchor_compiler"],
                 str(row["anchor_version"]), row["anchor_opt"],
                 row["anchor_arch"], row["anchor_function_file"])
        t_key = (row["target_function_name"], row["target_compiler"],
                 str(row["target_version"]), row["target_opt"],
                 row["target_arch"], row["target_function_file"])
        a_idx = self.mapping[a_key]
        t_idx = self.mapping[t_key]
        a_data = self.dataset[a_idx]
        t_data = self.dataset[t_idx]

        a_input_ids, a_adj = self.process_function(a_data)
        t_input_ids, t_adj = self.process_function(t_data)
        label = -1 if int(row["label"]) == 0 else 1
        label = torch.tensor(label, dtype=torch.float32)
        return a_input_ids, a_adj, t_input_ids, t_adj, label

    def graph_sizes(self):
        """Per-pair max node count (anchor/target), cached, for bucketing."""
        if getattr(self, "_graph_sizes", None) is None:
            sizes = []
            for row in self.df.itertuples(index=False):
                a_key = (row.anchor_function_name, row.anchor_compiler,
                         str(row.anchor_version), row.anchor_opt,
                         row.anchor_arch, row.anchor_function_file)
                t_key = (row.target_function_name, row.target_compiler,
                         str(row.target_version), row.target_opt,
                         row.target_arch, row.target_function_file)
                a_n = len(self.dataset[self.mapping[a_key]]["instruction_blocks"])
                t_n = len(self.dataset[self.mapping[t_key]]["instruction_blocks"])
                sizes.append(max(a_n, t_n))
            self._graph_sizes = sizes
        return self._graph_sizes

    def process_function(self, func_data):
        """Variable-size graph: no padding, no clipping (paper: CNN works on
        inputs with different sizes without padding/clipping)."""
        instr_blocks = func_data["instruction_blocks"]
        adj_data = func_data["adjacency_matrix"]
        adj = coo_matrix(
            (adj_data["data"], (adj_data["row"], adj_data["col"])),
            shape=adj_data["shape"],
        )
        processed_blocks = []
        for block in instr_blocks:
            enc = self.tokenizer(
                block,
                max_length=self.seq_len,
                padding="max_length",
                truncation=True,
            )
            processed_blocks.append(torch.tensor(enc["input_ids"], dtype=torch.long))
        input_ids = torch.stack(processed_blocks, dim=0)  # [N, L]
        return input_ids, adj  # sparse COO, native size


class Task2Dataset(Dataset):
    """Task 2 (paper): single-function samples labeled by optimization level
    (O0-O3), one dataset per platform (x64 / arm64)."""

    def __init__(self, function_list_path, dataset_path, function_idx_mapping_path,
                 tokenizer, seq_len=128, max_nodes=None):
        with open(function_list_path, "rb") as f:
            self.functions = pickle.load(f)  # (name, compiler, version, opt, arch, file)
        self.dataset = load_dataset(
            "json",
            data_files=dataset_path,
            split="train",
            cache_dir=os.path.join(".", "outputs", "cache"),
            keep_in_memory=False,
        )
        with open(function_idx_mapping_path, "rb") as f:
            self.mapping = pickle.load(f)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.opt_to_idx = {o: i for i, o in enumerate(opt)}

    def __len__(self):
        return len(self.functions)

    def __getitem__(self, idx):
        key = self.functions[idx]  # (name, compiler, version, opt, arch, file)
        data_idx = self.mapping[key]
        data = self.dataset[data_idx]
        input_ids, adj = self.process_function(data)
        label = self.opt_to_idx[key[3]]
        return input_ids, adj, torch.tensor(label, dtype=torch.long)

    def graph_sizes(self):
        """Per-sample node count, cached, for bucketing."""
        if getattr(self, "_graph_sizes", None) is None:
            self._graph_sizes = [
                len(self.dataset[self.mapping[k]]["instruction_blocks"])
                for k in self.functions
            ]
        return self._graph_sizes

    def process_function(self, func_data):
        """Variable-size graph: no padding, no clipping."""
        instr_blocks = func_data["instruction_blocks"]
        adj_data = func_data["adjacency_matrix"]
        adj = coo_matrix(
            (adj_data["data"], (adj_data["row"], adj_data["col"])),
            shape=adj_data["shape"],
        )
        processed_blocks = []
        for block in instr_blocks:
            enc = self.tokenizer(
                block,
                max_length=self.seq_len,
                padding="max_length",
                truncation=True,
            )
            processed_blocks.append(torch.tensor(enc["input_ids"], dtype=torch.long))
        input_ids = torch.stack(processed_blocks, dim=0)  # [N, L]
        return input_ids, adj  # sparse COO, native size
