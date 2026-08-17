"""Collate functions for the four BERT pre-training tasks.

Each collate fn takes a batch of raw function samples (as produced by
`FunctionDataset`) and returns a model-ready dict:
    {"task_type", "input_ids", "token_type_ids", "attention_mask", "labels"}

All tokenization goes through the HuggingFace tokenizer API, so
`token_type_ids` (segment ids) and `attention_mask` are generated
automatically.
"""
import random
import torch
from scipy.sparse import lil_matrix


def mask_tokens(input_ids, tokenizer, mlm_prob=0.15):
    """Standard BERT masking (80% <MASK> / 10% random / 10% unchanged).

    Special tokens are never masked. Labels are -100 at non-masked positions
    so they are ignored by cross entropy.
    """
    labels = input_ids.clone()
    eligible = torch.ones_like(input_ids, dtype=torch.bool)
    for sid in tokenizer.all_special_ids:
        eligible &= input_ids != sid

    prob = torch.rand(input_ids.shape)
    to_mask = eligible & (prob < mlm_prob)
    labels[~to_mask] = -100

    rand = torch.rand(input_ids.shape)
    input_ids[to_mask & (rand < 0.8)] = tokenizer.mask_token_id
    n_random = (to_mask & (rand >= 0.8) & (rand < 0.9)).sum().item()
    if n_random > 0:
        idx = (to_mask & (rand >= 0.8) & (rand < 0.9))
        input_ids[idx] = torch.randint(
            len(tokenizer.vocab), (n_random,), dtype=torch.long
        )
    return input_ids, labels


def encode_pair(tokenizer, texts_a, texts_b, seq_len):
    """Encode a list of block pairs into padded tensors (BERT pair format)."""
    enc = tokenizer(
        list(texts_a), list(texts_b),
        max_length=seq_len, padding="max_length", truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["token_type_ids"], enc["attention_mask"]


def encode_single(tokenizer, texts, seq_len):
    """Encode a list of blocks into padded tensors (BERT single format)."""
    enc = tokenizer(
        list(texts),
        max_length=seq_len, padding="max_length", truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["token_type_ids"], enc["attention_mask"]


def empty_batch(task_type):
    return {
        "task_type": task_type,
        "input_ids": torch.zeros(0, dtype=torch.long),
        "token_type_ids": torch.zeros(0, dtype=torch.long),
        "attention_mask": torch.zeros(0, dtype=torch.long),
        "labels": torch.zeros(0, dtype=torch.long),
    }


class MLMCollateFn:
    """Masked language modeling on the token sequence inside a single block
    (paper: "For the token sequences inside the node, we employ MLM")."""

    def __init__(self, tokenizer, seq_len=128, max_samples=40, train=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.train = train

    def __call__(self, batches):
        texts = []
        for batch in batches:
            blocks = batch["instruction_blocks"]
            if not blocks:
                continue
            n_samples = min(len(blocks), self.max_samples)
            texts.extend(random.sample(blocks, n_samples))
        if not texts:
            return empty_batch("mlm")
        input_ids, token_type_ids, attention_mask = encode_single(
            self.tokenizer, texts, self.seq_len
        )
        # Validation must use the same MLM objective as training. Feeding the
        # visible token while scoring it makes validation an identity task and
        # cannot be used to select a checkpoint.
        input_ids, labels = mask_tokens(input_ids, self.tokenizer)
        return {
            "task_type": "mlm",
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ANPCollateFn:
    """Adjacency node prediction: adjacent (1) vs non-adjacent (0) block pairs."""

    def __init__(self, tokenizer, seq_len=128, max_samples=40, train=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.train = train

    def __call__(self, batches):
        texts_a, texts_b, labels_list = [], [], []
        for batch in batches:
            blocks = batch["instruction_blocks"]
            adj = batch["adjacency_matrix"]
            n_blocks = len(blocks)
            shape = tuple(adj["shape"])
            adj_matrix = lil_matrix(shape, dtype=bool)
            adj_matrix[adj["row"], adj["col"]] = True

            positive_pairs = list(zip(adj["row"], adj["col"]))
            n_pos = len(positive_pairs)
            if n_pos == 0:
                continue

            # Negative pairs: random non-adjacent, same function.
            negative_pairs = set()
            attempts = 0
            while len(negative_pairs) < n_pos and attempts < n_pos * 20:
                attempts += 1
                i, j = random.sample(range(n_blocks), 2)
                if i != j and not adj_matrix[i, j] and (i, j) not in negative_pairs:
                    negative_pairs.add((i, j))
            negative_pairs = list(negative_pairs)

            total = min(2 * n_pos, self.max_samples) // 2 * 2
            half = total // 2
            pos_sample = random.sample(positive_pairs, min(half, n_pos))
            neg_sample = random.sample(negative_pairs, min(half, len(negative_pairs)))
            pairs = [(i, j, 1) for i, j in pos_sample] + [
                (i, j, 0) for i, j in neg_sample
            ]
            random.shuffle(pairs)
            for i, j, label in pairs:
                texts_a.append(blocks[i])
                texts_b.append(blocks[j])
                labels_list.append(label)
        if not texts_a:
            return empty_batch("anp")
        input_ids, token_type_ids, attention_mask = encode_pair(
            self.tokenizer, texts_a, texts_b, self.seq_len
        )
        return {
            "task_type": "anp",
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


class BIGCollateFn:
    """Block inside graph: same-function (1) vs cross-function (0) block pairs."""

    def __init__(self, tokenizer, seq_len=128, max_samples=40, train=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.train = train

    def __call__(self, batches):
        texts_a, texts_b, labels_list = [], [], []
        # Positive: two blocks sampled from the same function.
        func_blocks = []
        for batch in batches:
            blocks = batch["instruction_blocks"]
            if len(blocks) >= 2:
                func_blocks.append(blocks)
                n_pos = min(len(blocks) // 2, self.max_samples)
                seen = set()
                while len(seen) < n_pos:
                    i, j = random.sample(range(len(blocks)), 2)
                    if i == j or (i, j) in seen or (j, i) in seen:
                        continue
                    seen.add((i, j))
                    texts_a.append(blocks[i])
                    texts_b.append(blocks[j])
                    labels_list.append(1)
        # Negative: one block from each of two different functions.
        n_neg = len(labels_list)
        for _ in range(n_neg):
            if len(func_blocks) < 2:
                break
            f1, f2 = random.sample(range(len(func_blocks)), 2)
            b1 = random.choice(func_blocks[f1])
            b2 = random.choice(func_blocks[f2])
            texts_a.append(b1)
            texts_b.append(b2)
            labels_list.append(0)
        if not texts_a:
            return empty_batch("big")
        input_ids, token_type_ids, attention_mask = encode_pair(
            self.tokenizer, texts_a, texts_b, self.seq_len
        )
        return {
            "task_type": "big",
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


class GCCollateFn:
    """Graph classification: predict the (opt, arch) class of a block."""

    def __init__(self, tokenizer, seq_len=128, max_samples=40, train=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.train = train

    def __call__(self, batches):
        texts, labels_list = [], []
        for batch in batches:
            blocks = batch["instruction_blocks"]
            n_samples = min(len(blocks), self.max_samples)
            for block in random.sample(blocks, n_samples):
                texts.append(block)
                labels_list.append(batch["opt_arch_idx"])
        if not texts:
            return empty_batch("gc")
        input_ids, token_type_ids, attention_mask = encode_single(
            self.tokenizer, texts, self.seq_len
        )
        return {
            "task_type": "gc",
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


class CombinedCollateFn:
    """Dispatch to the task-specific collate fn based on `task_type`."""

    def __init__(self, mlm_collate=None, anp_collate=None,
                 big_collate=None, gc_collate=None):
        self.collates = {
            "mlm": mlm_collate,
            "anp": anp_collate,
            "big": big_collate,
            "gc": gc_collate,
        }

    def __call__(self, batches):
        task_type = batches[0]["task_type"]
        collate = self.collates[task_type]
        if collate is None:
            raise ValueError(f"Task {task_type} is not enabled")
        return collate(batches)


def sparse_pair_collate_fn(batch):
    """Collate for the fine-tuning pair dataset (variable-size graphs)."""
    a_ids, a_adj, t_ids, t_adj, labels = [], [], [], [], []
    for item in batch:
        if item[0] is None:
            continue
        a_ids.append(item[0])
        a_adj.append(item[1])
        t_ids.append(item[2])
        t_adj.append(item[3])
        labels.append(item[4])
    if not labels:
        return (torch.tensor([]), torch.tensor([]),
                torch.tensor([]), torch.tensor([]), torch.tensor([]))
    return (a_ids, a_adj, t_ids, t_adj, torch.stack(labels))
