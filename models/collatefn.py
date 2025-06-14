import numpy as np
import torch
import random
from models.dataset import BERTMLMDataset, BERTANPDataset
class MLMCollateFn:
    def __init__(self, tokenizer, seq_len=128, train=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.train = train

    def __call__(self, batches):
        ids_output = []
        labels_output = []
        for batch in batches:
            instruction_blocks = batch["instruction_blocks"]
            mlm_dataset = BERTMLMDataset(instruction_blocks, self.tokenizer, max_len=self.seq_len, train=self.train)
            for i in range(len(mlm_dataset)):
                ids, labels = mlm_dataset[i]
                ids_output.append(ids)
                labels_output.append(labels)
        # ids_output 和 labels_output 都是列表，转换为张量
        ids_output = torch.stack(ids_output, dim=0)
        labels_output = torch.stack(labels_output, dim=0)
        return ids_output, labels_output
class ANPCollateFn:
    def __init__(self, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __call__(self, batches):
        ids_a_output = []
        ids_b_output = []
        labels_output = []
        for batch in batches:
            instruction_blocks = batch["instruction_blocks"]
            adj = batch["adjacency_matrix"]
            anp_dataset = BERTANPDataset(instruction_blocks, self.tokenizer, adj, max_len=self.seq_len)
            for i in range(len(anp_dataset)):
                ids_a, ids_b, label = anp_dataset[i]
                ids_a_output.append(ids_a)
                ids_b_output.append(ids_b)
                labels_output.append(label)
        # ids_a_output, ids_b_output 和 labels_output 都是列表，转换为张量
        ids_a_output = torch.stack(ids_a_output, dim=0)
        ids_b_output = torch.stack(ids_b_output, dim=0)
        labels_output = torch.tensor(labels_output, dtype=torch.long)
        return ids_a_output, ids_b_output, labels_output

class MLM_ANP_CollateFn:
    def __init__(self, tokenizer, seq_len=128, train=True):
        self.mlm_collate_fn = MLMCollateFn(tokenizer, seq_len, train)
        self.anp_collate_fn = ANPCollateFn(tokenizer, seq_len)

    def __call__(self, batches):
        mlm_ids, mlm_labels = self.mlm_collate_fn(batches)
        anp_ids_a, anp_ids_b, anp_labels = self.anp_collate_fn(batches)
        return {
            "mlm_input_ids": mlm_ids,
            "mlm_labels": mlm_labels,
            "anp_input_ids_a": anp_ids_a,
            "anp_input_ids_b": anp_ids_b,
            "anp_labels": anp_labels
        }