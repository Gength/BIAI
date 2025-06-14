import numpy as np
import torch
import random
from models.dataset import BERTMLMDataset, BERTANPDataset, TaskDataset
class MLMCollateFn:
    def __init__(self, tokenizer, seq_len=128, train=True, samples_per_batch=20):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.train = train
        self.samples_per_batch = samples_per_batch

    def __call__(self, batches):
        ids_output = []
        labels_output = []
        for batch in batches:
            instruction_blocks = batch["instruction_blocks"]
            mlm_dataset = BERTMLMDataset(instruction_blocks, self.tokenizer, max_len=self.seq_len, train=self.train)
            # Only take one sample per graph to avoid data imbalance
            if len(mlm_dataset) > 0:
                if len(mlm_dataset) < self.samples_per_batch:
                    # If dataset is smaller than samples_per_batch, take all samples
                    idx = range(len(mlm_dataset))
                else:
                    # Randomly select samples from the dataset
                    idx = random.sample(range(len(mlm_dataset)), self.samples_per_batch)
                idx = random.randint(0, len(mlm_dataset)-1)
                ids, labels = mlm_dataset[idx]
                ids_output.append(ids)
                labels_output.append(labels)
        # Handle empty batch case
        if len(ids_output) == 0:
            return torch.tensor([]), torch.tensor([])

        # ids_output and labels_output are lists, convert to tensors
        ids_output = torch.stack(ids_output, dim=0)
        labels_output = torch.stack(labels_output, dim=0)
        return {
            'task_type': 'mlm',
            "input_ids": ids_output,
            "labels": labels_output
        }

class ANPCollateFn:
    def __init__(self, tokenizer, seq_len=128, samples_per_batch=20):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples_per_batch = samples_per_batch

    def __call__(self, batches):
        ids_a_output = []
        ids_b_output = []
        labels_output = []
        for batch in batches:
            instruction_blocks = batch["instruction_blocks"]
            adj = batch["adjacency_matrix"]
            anp_dataset = BERTANPDataset(instruction_blocks, self.tokenizer, adj, max_len=self.seq_len)
            # Only take one sample per graph to avoid data imbalance
            if len(anp_dataset) > 0:
                if len(anp_dataset) < self.samples_per_batch:
                    # If dataset is smaller than samples_per_batch, take all samples
                    idx = range(len(anp_dataset))
                else:
                    # Randomly select samples from the dataset
                    idx = random.sample(range(len(anp_dataset)), self.samples_per_batch)
                ids_a, ids_b, label = anp_dataset[idx]
                ids_a_output.append(ids_a)
                ids_b_output.append(ids_b)
                labels_output.append(label)
        # Handle empty batch case
        if len(ids_a_output) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        # ids_a_output, ids_b_output and labels_output are lists, convert to tensors
        ids_a_output = torch.stack(ids_a_output, dim=0)
        ids_b_output = torch.stack(ids_b_output, dim=0)
        labels_output = torch.tensor(labels_output, dtype=torch.long)
        return {
            'task_type': 'anp',
            "input_a": ids_a_output,
            "input_b": ids_b_output,
            "labels": labels_output
        }

class CombinedCollateFn:
    """Select different collate functions according to task type"""
    def __init__(self, mlm_collate, anp_collate):
        self.mlm_collate = mlm_collate
        self.anp_collate = anp_collate
    
    def __call__(self, batches):
        # All batches should have the same task type
        task_type = batches[0]["task_type"]
        
        if task_type == "mlm":
            return self.mlm_collate(batches)
        elif task_type == "anp":
            return self.anp_collate(batches)
        else:
            raise ValueError(f"Unknown task type: {task_type}")