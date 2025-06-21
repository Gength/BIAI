import torch
import random
from models.dataset import BERTANPDataset
from models.tokenizer import AsmTokenizer
class MLMCollateFn:
    def __init__(self, tokenizer: AsmTokenizer, seq_len=128, train=True, samples_per_batch=10):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.train = train
        self.samples_per_batch = samples_per_batch

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
            
            # At least 2 blocks are needed to form sample pairs
            if n_blocks < 2:
                continue
                
            # Calculate the actual number of blocks that can be sampled
            n_blocks_needed = min(n_blocks, self.samples_per_batch + 1)
            
            # Randomly select the starting index
            start_idx = random.randint(0, n_blocks - n_blocks_needed) if n_blocks > n_blocks_needed else 0
            sampled_blocks = instruction_blocks[start_idx:start_idx + n_blocks_needed]
            
            # Create samples for each pair of adjacent blocks
            for i in range(len(sampled_blocks) - 1):
                # Concatenate the current block and the next block, add special tokens
                text = "<CLS> " + sampled_blocks[i] + " <SEP> " + sampled_blocks[i+1]
                ids = self.tokenizer.encode(text)
                
                # Apply mask (if in training mode)
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
        
        # Handle empty batch case
        if len(ids_output) == 0:
            return {
                'task_type': 'mlm',
                "input_ids": torch.tensor([]),
                "labels": torch.tensor([])
            }
        
        # Stack all samples
        ids_output = torch.stack(ids_output, dim=0)
        labels_output = torch.stack(labels_output, dim=0)
        
        return {
            'task_type': 'mlm',
            "input_ids": ids_output,
            "labels": labels_output
        }

class ANPCollateFn:
    def __init__(self, tokenizer, seq_len=128, samples_per_batch=10):
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
            if len(anp_dataset) > 0:
                n_samples = min(len(anp_dataset), self.samples_per_batch)
                half = len(anp_dataset) // 2
                n_first = n_samples // 2
                n_second = n_samples - n_first
                idx_first = random.sample(range(half), min(n_first, half))
                idx_second = random.sample(range(half, len(anp_dataset)), min(n_second, len(anp_dataset) - half))
                idx_set = idx_first + idx_second
                random.shuffle(idx_set)
                for idx in idx_set:
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
