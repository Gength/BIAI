import torch
import random
from models.tokenizer import AsmTokenizer
from utils.utility import random_mask, pad_sequence
from scipy.sparse import lil_matrix
import numpy as np
class MLMCollateFn:
    def __init__(self, tokenizer: AsmTokenizer, seq_len=128, max_samples=40, train=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.train = train
        self.max_samples = max_samples

    def __call__(self, batches):
        ids_output = []
        segment_ids_output = []
        labels_output = []
        
        for batch in batches:
            instruction_blocks = batch["instruction_blocks"]
            n_blocks = len(instruction_blocks)
            # Calculate the number of available block pairs (total consecutive block pairs)
            available_pairs = n_blocks - 1
                
            # Determine the actual number of samples (not exceeding available pairs and max_samples)
            n_samples = min(available_pairs, self.max_samples)
            
            # Generate all possible start indices (0 to n_blocks-2)
            all_start_indices = list(range(available_pairs))
            # Directly sample non-repeating start indices
            sampled_start_indices = random.sample(all_start_indices, n_samples)
            random.shuffle(sampled_start_indices)  # Shuffle the order
            
            for start_idx in sampled_start_indices:
                text = "<CLS> " + instruction_blocks[start_idx] + " <SEP> " + instruction_blocks[start_idx+1]
                segment_id = [0] * (len(instruction_blocks[start_idx]) + 1) + [1] * (len(instruction_blocks[start_idx+1]) + 1)
                ids = self.tokenizer.encode(text)
                
                if self.train:
                    ids, labels = random_mask(ids, self.tokenizer)
                else:
                    labels = ids.copy()
                
                # Truncate and pad
                ids = ids[:self.seq_len]
                labels = labels[:self.seq_len]
                segment_id = segment_id[:self.seq_len]
                ids = pad_sequence(ids, self.seq_len, self.tokenizer.pad_token_id)
                segment_id = pad_sequence(segment_id, self.seq_len, 0)  # Pad segment IDs
                labels = pad_sequence(labels, self.seq_len, 0) # Note: label for padding part should be 0 (ignore loss)
                
                ids_output.append(torch.tensor(ids, dtype=torch.int))
                segment_ids_output.append(torch.tensor(segment_id, dtype=torch.int))
                labels_output.append(torch.tensor(labels, dtype=torch.long))
        
        return {
            'task_type': 'mlm',
            "input_ids": torch.stack(ids_output, dim=0),
            "segment_ids": torch.stack(segment_ids_output, dim=0),
            "labels": torch.stack(labels_output, dim=0)
        }

class ANPCollateFn:
    def __init__(self, tokenizer, seq_len=128, max_samples=40, train=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.train = train

    def __call__(self, batches):
        ids_output = []
        labels_output = []
        segment_ids_output = []
        
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
            n_blocks = shape[0]  # Get the number of basic blocks
            
            # Handle the case with no positive pairs (n_pos == 0)
            if n_pos == 0:
                # Calculate the maximum possible number of negative pairs (avoid self-loops)
                max_neg_pairs = n_blocks * (n_blocks - 1)
                n_neg_samples = min(self.max_samples, max_neg_pairs)
                
                # Directly generate non-repeating negative pairs
                negative_pairs = set()
                while len(negative_pairs) < n_neg_samples:
                    i, j = random.sample(range(n_blocks), 2)
                    # Ensure not a self-loop and is a new negative pair
                    if i != j and (i, j) not in negative_pairs:
                        negative_pairs.add((i, j))
                
                all_pairs = list(negative_pairs)
                pair_labels = [0] * len(negative_pairs)  # All labels are 0 (negative pairs)
            
            # Handle the case with positive pairs
            else:
                # Generate negative pairs (same number as positive pairs)
                negative_pairs = set()
                while len(negative_pairs) < n_pos:
                    i, j = random.sample(range(n_blocks), 2)
                    # Ensure not a self-loop, no edge, and is a new negative pair
                    if i != j and adj_matrix[i, j] == 0 and (i, j) not in negative_pairs:
                        negative_pairs.add((i, j))
                
                negative_pairs = list(negative_pairs)
                
                # Calculate balanced sampling size
                total_samples = min(2 * n_pos, self.max_samples)
                total_samples = total_samples // 2 * 2  # Ensure even number
                half_samples = total_samples // 2
                
                # Sample positive and negative pairs
                pos_sample = random.sample(positive_pairs, min(half_samples, n_pos))
                neg_sample = random.sample(negative_pairs, min(half_samples, n_pos))
                
                all_pairs = pos_sample + neg_sample
                pair_labels = [1] * len(pos_sample) + [0] * len(neg_sample)
                
                # Shuffle the order
                combined = list(zip(all_pairs, pair_labels))
                random.shuffle(combined)
                all_pairs, pair_labels = zip(*combined) if combined else ([], [])
            
            # Process each block pair
            for (i, j), label in zip(all_pairs, pair_labels):
                # Format and encode text
                text = "<CLS> " + instruction_blocks[i] + " <SEP> " + instruction_blocks[j]
                segment_id = [0] * (len(instruction_blocks[i]) + 1) + [1] * (len(instruction_blocks[j]) + 1)
                ids = self.tokenizer.encode(text)
                
                # Apply random mask during training
                if self.train:
                    ids, _ = random_mask(ids, self.tokenizer)
                
                # Truncate and pad
                ids = ids[:self.seq_len]
                segment_id = segment_id[:self.seq_len]
                ids = pad_sequence(ids, self.seq_len, self.tokenizer.pad_token_id)
                segment_id = pad_sequence(segment_id, self.seq_len, 0)  # Pad segment IDs
                ids_tensor = torch.tensor(ids, dtype=torch.int)
                label_tensor = torch.tensor(label, dtype=torch.long)
                
                ids_output.append(ids_tensor)
                segment_ids_output.append(torch.tensor(segment_id, dtype=torch.int))
                labels_output.append(label_tensor)
        
        # Handle empty batch
        if not ids_output:
            return {
                'task_type': 'anp',
                "input_ids": torch.tensor([], dtype=torch.int),
                "segment_ids": torch.tensor([], dtype=torch.int),
                "labels": torch.tensor([], dtype=torch.long)
            }
        
        return {
            'task_type': 'anp',
            "input_ids": torch.stack(ids_output),
            "segment_ids": torch.stack(segment_ids_output),
            "labels": torch.stack(labels_output)
        }

class BIGCollateFn:
    def __init__(self, tokenizer, seq_len=128, max_samples=40, train=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
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
            n_pos_samples = min(n_blocks//2, self.max_samples)
            n_pos_sample_idx = set()
            while len(n_pos_sample_idx) < n_pos_samples:
                # Randomly select two different blocks
                i, j = random.sample(range(n_blocks), 2)
                if i == j or (i, j) in n_pos_sample_idx or (j, i) in n_pos_sample_idx:
                    continue
                n_pos_sample_idx.add((i, j))
            
            for i,j in n_pos_sample_idx:
                block1 = instruction_blocks[i]
                block2 = instruction_blocks[j]
                
                # Build input sequence: <CLS> block1 <SEP> block2
                text = "<CLS> " + block1 + " <SEP> " + block2
                segment_ids = [0] * (len(block1) + 1) + [1] * (len(block2) + 1)

                ids = self.tokenizer.encode(text)
                
                # Apply random mask during training
                if self.train:
                    masked_ids, _ = random_mask(ids, self.tokenizer)
                else:
                    masked_ids = ids
                
                # Truncate and pad
                masked_ids = masked_ids[:self.seq_len]
                segment_ids = segment_ids[:self.seq_len]
                masked_ids = pad_sequence(masked_ids, self.seq_len, self.tokenizer.pad_token_id)
                segment_ids = pad_sequence(segment_ids, self.seq_len, 0)  # Pad segment IDs
                
                # Add as positive sample (label 1)
                samples.append((masked_ids, segment_ids, 1))
        
        # Generate negative samples (block pairs from different functions)
        n_neg_samples = len(samples)  # Match number of positive samples
        # The probability of sampling the same block pair from the same function twice is very low, so deduplication is not performed
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
            segment_ids = [0] * (len(block1) + 1) + [1] * (len(block2) + 1)
            
            # Apply random mask during training
            if self.train:
                masked_ids, _ = random_mask(ids, self.tokenizer)
            else:
                masked_ids = ids
            
            # Truncate and pad
            masked_ids = masked_ids[:self.seq_len]
            segment_ids = segment_ids[:self.seq_len]
            masked_ids = pad_sequence(masked_ids, self.seq_len, self.tokenizer.pad_token_id)
            segment_ids = pad_sequence(segment_ids, self.seq_len, 0)  # Pad segment IDs
            
            # Add as negative sample (label 0)
            samples.append((masked_ids, segment_ids, 0))
        
        # Shuffle all samples together
        random.shuffle(samples)
        
        # Handle empty batch
        if len(samples) == 0:
            return {
                'task_type': 'big',
                "input_ids": torch.tensor([]),
                "segment_ids": torch.tensor([]),
                "labels": torch.tensor([])
            }
        
        # Unzip into separate lists
        input_ids_list, segment_ids_list, labels_list = zip(*samples)
        
        return {
            'task_type': 'big',
            "input_ids": torch.stack([torch.tensor(x, dtype=torch.int) for x in input_ids_list]),
            "segment_ids": torch.stack([torch.tensor(x, dtype=torch.int) for x in segment_ids_list]),
            "labels": torch.tensor(labels_list, dtype=torch.long)
        }

class GCCollateFn:
    def __init__(self, tokenizer, seq_len=128, max_samples=40):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples

    def __call__(self, batches):
        input_ids_list = []
        labels_list = []
        segment_ids_list = []
        
        for batch in batches:
            instruction_blocks = batch["instruction_blocks"]
            opt_arch_idx = batch["opt_arch_idx"]
            n_blocks = len(instruction_blocks)
            
            # Determine sample size
            n_samples = min(n_blocks, self.max_samples)
            sampled_blocks = random.sample(instruction_blocks, min(n_samples, n_blocks))
            
            for block in sampled_blocks:
                text = "<CLS> " + block
                ids = self.tokenizer.encode(text)
                segment_ids = [0] * (len(block) + 1)
                ids = ids[:self.seq_len]
                segment_ids = segment_ids[:self.seq_len]
                ids = pad_sequence(ids, self.seq_len, self.tokenizer.pad_token_id)
                segment_ids = pad_sequence(segment_ids, self.seq_len, 0)  # Pad segment IDs
                input_ids_list.append(torch.tensor(ids, dtype=torch.int))
                segment_ids_list.append(torch.tensor(segment_ids, dtype=torch.int))
                labels_list.append(opt_arch_idx)
 
        if len(input_ids_list) == 0:
            return {
                'task_type': 'gc',
                "input_ids": torch.tensor([]),
                "segment_ids": torch.tensor([]),
                "labels": torch.tensor([])
            }
        
        input_ids_tensor = torch.stack(input_ids_list, dim=0)
        segment_ids_tensor = torch.stack(segment_ids_list, dim=0)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        
        return {
            'task_type': 'gc',
            "input_ids": input_ids_tensor,
            "segment_ids": segment_ids_tensor,
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
