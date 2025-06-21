import torch
import random
from models.dataset import BERTANPDataset
from models.tokenizer import AsmTokenizer
class MLMCollateFn:
    def __init__(self, tokenizer: AsmTokenizer, seq_len=128, train=True, min_samples=20, max_samples=80, coverage_ratio=0.7):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.train = train
        self.min_samples = min_samples  # Minimum samples for small functions
        self.max_samples = max_samples  # Maximum samples for large functions
        self.coverage_ratio = coverage_ratio  # Target coverage ratio

    def random_mask(self, ids):
        output = []
        labels = []
        for token_id in ids:
            if token_id == self.tokenizer.sep_token_id or token_id == self.tokenizer.cls_token_id:
                output.append(token_id)
                labels.append(0)
            elif random.random() < 0.15:
                rand_val = random.random()
                if rand_val < 0.8:
                    output.append(self.tokenizer.mask_token_id)  # 80% replaced with MASK
                elif rand_val < 0.9:
                    output.append(random.choice(list(self.tokenizer.vocab.values())))  # 10% random token
                else:
                    output.append(token_id)  # 10% keep original
                labels.append(token_id)
            else:
                output.append(token_id)
                labels.append(0)
        return output, labels

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
                        ids, labels = self.random_mask(ids)
                    else:
                        labels = ids.copy()
                    
                    # Truncate and pad
                    ids = ids[:self.seq_len]
                    labels = labels[:self.seq_len]
                    pad_len = self.seq_len - len(ids)
                    if pad_len > 0:
                        ids += [self.tokenizer.pad_token_id] * pad_len
                        labels += [0] * pad_len
                    
                    ids_output.append(torch.tensor(ids, dtype=torch.long))
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
    def __init__(self, tokenizer, seq_len=128, min_samples=15, max_samples=60, neg_ratio=1.0):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.neg_ratio = neg_ratio  # Negative sample ratio

    def __call__(self, batches):
        ids_output = []
        labels_output = []
        
        for batch in batches:
            instruction_blocks = batch["instruction_blocks"]
            adj = batch["adjacency_matrix"]
            anp_dataset = BERTANPDataset(
                instruction_blocks, 
                self.tokenizer, 
                adj, 
                max_len=self.seq_len
            )
            
            if len(anp_dataset) > 0:
                # Dynamically calculate the number of samples
                n_samples = min(
                    max(self.min_samples, int(len(anp_dataset) * 0.3)),
                    self.max_samples
                )
                
                # Stratified sampling strategy
                positive_count = min(len(anp_dataset) // 2, n_samples // 2)
                negative_count = min(
                    len(anp_dataset) - len(anp_dataset) // 2,
                    int(positive_count * self.neg_ratio)
                )
                
                # Ensure the total number of samples does not exceed the limit
                total_samples = min(positive_count + negative_count, n_samples)
                
                # Sample positive and negative samples
                idx_first = random.sample(
                    range(len(anp_dataset) // 2), 
                    min(positive_count, len(anp_dataset) // 2)
                )
                idx_second = random.sample(
                    range(len(anp_dataset) // 2, len(anp_dataset)), 
                    min(negative_count, len(anp_dataset) - len(anp_dataset) // 2)
                )
                idx_set = idx_first + idx_second
                random.shuffle(idx_set)
                
                for idx in idx_set[:total_samples]:
                    ids, label = anp_dataset[idx]
                    ids_output.append(ids)
                    labels_output.append(label)
        
        if len(ids_output) == 0:
            return {
                'task_type': 'anp',
                "input_ids": torch.tensor([]),
                "labels": torch.tensor([])
            }
        
        return {
            'task_type': 'anp',
            "input_ids": torch.stack(ids_output, dim=0),
            "labels": torch.tensor(labels_output, dtype=torch.long)
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
