import torch
import random
from models.tokenizer import AsmTokenizer
from utils.utility import random_mask
from scipy.sparse import lil_matrix
import numpy as np
class MLMCollateFn:
    def __init__(self, tokenizer: AsmTokenizer, seq_len=128, train=True, min_samples=20, max_samples=80, coverage_ratio=0.7):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.train = train
        self.min_samples = min_samples  # Minimum samples for small functions
        self.max_samples = max_samples  # Maximum samples for large functions
        self.coverage_ratio = coverage_ratio  # Target coverage ratio

    def __call__(self, batches):
        ids_output = []
        labels_output = []
        
        for batch in batches:
            instruction_blocks = batch["instruction_blocks"]
            n_blocks = len(instruction_blocks)
            
            # Dynamically calculate the number of samples (based on the number of blocks)
            if n_blocks <= 30:  # Small function
                n_samples = min(n_blocks - 1, self.max_samples)
            elif n_blocks <= 100:  # Medium function
                n_samples = min(
                    max(self.min_samples, int(n_blocks * self.coverage_ratio)),
                    self.max_samples
                )
            else:  # Large function
                n_samples = self.max_samples
            
            # Stratified sampling strategy
            sampled_pairs = set()
            for _ in range(n_samples):
                # Randomly select start position
                start_idx = random.randint(0, n_blocks - 2)
                pair = (start_idx, start_idx + 1)
                
                # Ensure not to sample the same block pair repeatedly
                if pair not in sampled_pairs:
                    sampled_pairs.add(pair)
                    text = "<CLS> " + instruction_blocks[start_idx] + " <SEP> " + instruction_blocks[start_idx+1]
                    ids = self.tokenizer.encode(text)
                    
                    if self.train:
                        ids, labels = random_mask(ids, self.tokenizer)
                    else:
                        labels = ids.copy()
                    
                    # Truncate and pad
                    ids = ids[:self.seq_len]
                    labels = labels[:self.seq_len]
                    pad_len = self.seq_len - len(ids)
                    if pad_len > 0:
                        ids += [self.tokenizer.pad_token_id] * pad_len
                        labels += [0] * pad_len
                    
                    ids_output.append(torch.tensor(ids, dtype=torch.int))
                    labels_output.append(torch.tensor(labels, dtype=torch.long))
        
        # Handle empty batch
        if len(ids_output) == 0:
            return {
                'task_type': 'mlm',
                "input_ids": torch.tensor([]),
                "labels": torch.tensor([])
            }
        
        return {
            'task_type': 'mlm',
            "input_ids": torch.stack(ids_output, dim=0),
            "labels": torch.stack(labels_output, dim=0)
        }

class ANPCollateFn:
    def __init__(self, tokenizer, seq_len=128, min_samples=15, max_samples=60, train=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.train = train

    def __call__(self, batches):
        ids_output = []
        labels_output = []
        
        for batch in batches:
            instruction_blocks = batch["instruction_blocks"]
            adj = batch["adjacency_matrix"]
            
            # Construct adjacency matrix
            shape = tuple(adj['shape'])
            adj_matrix = lil_matrix(shape, dtype=np.int32)
            adj_matrix[adj['row'], adj['col']] = adj['data']
            
            # Extract positive pairs
            positive_pairs = list(zip(adj['row'], adj['col']))
            n_pos = len(positive_pairs)
            
            # Skip function with no positive pairs
            if n_pos == 0:
                continue
                
            # Generate negative pairs (equal count to positive pairs)
            block_ids = list(range(shape[0]))
            negative_pairs = []
            while len(negative_pairs) < n_pos:
                i, j = random.sample(block_ids, 2)
                if adj_matrix[i, j] == 0:  # Ensure no edge exists
                    negative_pairs.append((i, j))
            
            # Calculate balanced sample count
            total_samples = min(max(self.min_samples, 2 * n_pos), self.max_samples)
            total_samples = total_samples // 2 * 2  # Ensure even number
            n_samples = min(total_samples, 2 * n_pos)
            half_samples = n_samples // 2
            
            # Sample positive and negative pairs
            pos_sample = random.sample(positive_pairs, min(half_samples, n_pos))
            neg_sample = random.sample(negative_pairs, min(half_samples, n_pos))
            all_pairs = pos_sample + neg_sample
            pair_labels = [1] * len(pos_sample) + [0] * len(neg_sample)
            
            # Shuffle pairs and labels together
            combined = list(zip(all_pairs, pair_labels))
            random.shuffle(combined)
            all_pairs, pair_labels = zip(*combined) if combined else ([], [])
            
            # Process each pair
            for (i, j), label in zip(all_pairs, pair_labels):
                # Format and tokenize instructions
                text = "<CLS> " + instruction_blocks[i] + " <SEP> " + instruction_blocks[j]
                ids = self.tokenizer.encode(text)
                
                # Apply masking during training
                if self.train:
                    ids, _ = random_mask(ids, self.tokenizer)
                
                # Truncate and pad sequences
                ids = ids[:self.seq_len]
                padding = [self.tokenizer.pad_token_id] * (self.seq_len - len(ids))
                ids_tensor = torch.tensor(ids + padding, dtype=torch.int)
                label_tensor = torch.tensor(label, dtype=torch.long)
                
                ids_output.append(ids_tensor)
                labels_output.append(label_tensor)
        
        # Handle empty batch case
        if not ids_output:
            return {
                'task_type': 'anp',
                "input_ids": torch.tensor([], dtype=torch.int),
                "labels": torch.tensor([], dtype=torch.long)
            }
        
        return {
            'task_type': 'anp',
            "input_ids": torch.stack(ids_output),
            "labels": torch.stack(labels_output)
        }

class BIGCollateFn:
    def __init__(self, tokenizer, seq_len=128, min_samples=15, max_samples=60, train=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.train = train

    def __call__(self, batches):
        # Collect all samples in a unified list
        samples = []
        
        # Generate positive samples (block pairs within the same function)
        for batch in batches:
            instruction_blocks = batch["instruction_blocks"]
            n_blocks = len(instruction_blocks)
            
            if n_blocks < 2:
                continue  # Skip functions with only one block
                
            # Determine the number of samples
            n_samples = min(max(self.min_samples, n_blocks//2), self.max_samples)
            
            for _ in range(n_samples):
                # Randomly select two different blocks
                i, j = random.sample(range(n_blocks), 2)
                block1 = instruction_blocks[i]
                block2 = instruction_blocks[j]
                
                # Build input sequence: <CLS> block1 <SEP> block2
                text = "<CLS> " + block1 + " <SEP> " + block2
                ids = self.tokenizer.encode(text)
                
                # Apply random mask during training
                if self.train:
                    masked_ids, _ = random_mask(ids, self.tokenizer)
                else:
                    masked_ids = ids
                
                # Truncate and pad
                masked_ids = masked_ids[:self.seq_len]
                pad_len = self.seq_len - len(masked_ids)
                if pad_len > 0:
                    masked_ids += [self.tokenizer.pad_token_id] * pad_len
                
                # Add as positive sample (label 1)
                samples.append((torch.tensor(masked_ids, dtype=torch.long), 1))
        
        # Generate negative samples (block pairs from different functions)
        n_neg_samples = len(samples)  # Match number of positive samples
        for _ in range(n_neg_samples):
            # Randomly select two different functions
            func_idx1, func_idx2 = random.sample(range(len(batches)), 2)
            func1_blocks = batches[func_idx1]["instruction_blocks"]
            func2_blocks = batches[func_idx2]["instruction_blocks"]
            
            # Skip if either function has no blocks
            if len(func1_blocks) == 0 or len(func2_blocks) == 0:
                continue
                
            # Randomly select one block from each function
            block1 = random.choice(func1_blocks)
            block2 = random.choice(func2_blocks)
            
            # Build input sequence: <CLS> block1 <SEP> block2
            text = "<CLS> " + block1 + " <SEP> " + block2
            ids = self.tokenizer.encode(text)
            
            # Apply random mask during training
            if self.train:
                masked_ids, _ = random_mask(ids, self.tokenizer)
            else:
                masked_ids = ids
            
            # Truncate and pad
            masked_ids = masked_ids[:self.seq_len]
            pad_len = self.seq_len - len(masked_ids)
            if pad_len > 0:
                masked_ids += [self.tokenizer.pad_token_id] * pad_len
            
            # Add as negative sample (label 0)
            samples.append((torch.tensor(masked_ids, dtype=torch.int), 0))
        
        # Shuffle all samples together
        random.shuffle(samples)
        
        # Handle empty batch
        if len(samples) == 0:
            return {
                'task_type': 'big',
                "input_ids": torch.tensor([]),
                "labels": torch.tensor([])
            }
        
        # Unzip into separate lists
        input_ids_list, labels_list = zip(*samples)
        
        return {
            'task_type': 'big',
            "input_ids": torch.stack(input_ids_list, dim=0),
            "labels": torch.tensor(labels_list, dtype=torch.long)
        }

class GCCollateFn:
    def __init__(self, tokenizer, seq_len=128, min_samples=10, max_samples=40):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_samples = min_samples
        self.max_samples = max_samples

    def __call__(self, batches):
        input_ids_list = []
        labels_list = []
        
        for batch in batches:
            instruction_blocks = batch["instruction_blocks"]
            opt_arch_idx = batch["opt_arch_idx"]
            n_blocks = len(instruction_blocks)
            
            # Determine sample size
            n_samples = min(max(self.min_samples, n_blocks), self.max_samples)
            sampled_blocks = random.sample(instruction_blocks, min(n_samples, n_blocks))
            
            for block in sampled_blocks:
                text = "<CLS> " + block
                ids = self.tokenizer.encode(text)
                ids = ids[:self.seq_len]
                pad_len = self.seq_len - len(ids)
                if pad_len > 0:
                    ids += [self.tokenizer.pad_token_id] * pad_len
                input_ids_list.append(torch.tensor(ids, dtype=torch.int))
                labels_list.append(opt_arch_idx)
 
        if len(input_ids_list) == 0:
            return {
                'task_type': 'gc',
                "input_ids": torch.tensor([]),
                "labels": torch.tensor([])
            }
        
        input_ids_tensor = torch.stack(input_ids_list, dim=0)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        
        return {
            'task_type': 'gc',
            "input_ids": input_ids_tensor,
            "labels": labels_tensor
        }
class CombinedCollateFn:
    def __init__(self, mlm_collate, anp_collate=None, big_collate=None, gc_collate=None):
        self.mlm_collate = mlm_collate
        self.anp_collate = anp_collate
        self.big_collate = big_collate
        self.gc_collate = gc_collate
    
    def __call__(self, batches):
        task_type = batches[0]["task_type"]
        
        if task_type == "mlm":
            return self.mlm_collate(batches)
        elif task_type == "anp":
            return self.anp_collate(batches)
        elif task_type == "big":
            return self.big_collate(batches)
        elif task_type == "gc":
            return self.gc_collate(batches)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

def sparse_collate_fn(batch):
    a_ids_list, a_idx_list, a_val_list = [], [], []
    t_ids_list, t_idx_list, t_val_list = [], [], []
    labels_list = []
    
    for item in batch:
        a_ids, a_idx, a_val, t_ids, t_idx, t_val, label = item
        a_ids_list.append(a_ids)
        a_idx_list.append(a_idx)
        a_val_list.append(a_val)
        t_ids_list.append(t_ids)
        t_idx_list.append(t_idx)
        t_val_list.append(t_val)
        labels_list.append(label)
    
    return (
        torch.stack(a_ids_list),
        a_idx_list,  # index list
        a_val_list,  # value list
        torch.stack(t_ids_list),
        t_idx_list,  # index list
        t_val_list,  # value list
        torch.stack(labels_list)
    )
