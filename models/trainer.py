"""Trainers for BERT pre-training (4 tasks) and siamese fine-tuning.

Pre-training: fixed learning rate 1e-4 (paper setting), optional linear
warmup + cosine decay, AMP on CUDA, wandb optional.
Fine-tuning: siamese CosineEmbeddingLoss (paper task 1 setup).
"""
import math
import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers.optimization import get_linear_schedule_with_warmup

from models.collatefn import (
    MLMCollateFn, ANPCollateFn, BIGCollateFn, GCCollateFn,
    CombinedCollateFn, sparse_pair_collate_fn,
)
from models.dataset import TaskDataset
from models.retrieval import evaluate_retrieval, load_function_keys


def _resolve_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print(f"[trainer] CUDA requested but unavailable, falling back to CPU")
        return torch.device("cpu")
    return dev


class BucketBatchSampler:
    """Group samples by graph size for stable native-graph memory usage.

    Samples are bucketed by log-size; each batch is drawn from one bucket.
    """

    def __init__(self, sizes, batch_size, shuffle=True, num_buckets=8, seed=42):
        import math
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        # Bucket id = floor(log2(max(1, size))), capped at num_buckets-1
        max_log = max(1, int(math.log2(max(1, max(sizes)))))
        buckets = {}
        for i, s in enumerate(sizes):
            bid = min(int(math.log2(max(1, s))), num_buckets - 1)
            buckets.setdefault(bid, []).append(i)
        self.buckets = [buckets.get(b, []) for b in range(num_buckets)]
        self._len = sum(
            max(1, len(b) // batch_size + (1 if len(b) % batch_size else 0))
            for b in self.buckets if b
        )

    def __iter__(self):
        batches = []
        for bucket in self.buckets:
            if not bucket:
                continue
            order = list(bucket)
            if self.shuffle:
                self.rng.shuffle(order)
            for i in range(0, len(order), self.batch_size):
                batches.append(order[i:i + self.batch_size])
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self._len


class BERTPretrainTrainer:
    """Train `BERTForPretraining` on the four tasks (MLM/ANP/BIG/GC)."""

    def __init__(self, model, tokenizer, config):
        self.config = config
        self.device = _resolve_device(config.device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
        self.scheduler = None  # created in train() once the dataset size is known

        self.use_amp = getattr(config, "use_amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.loss_weights = getattr(config, "loss_weights", None) or {
            "mlm": 1.0, "anp": 1.0, "big": 1.0, "gc": 1.0,
        }
        self.enabled_tasks = [t for t in ("mlm", "anp", "big", "gc")
                              if getattr(config, f"train_{t}", True)]

        # wandb (offline by default, like the previous setup)
        self.use_wandb = getattr(config, "use_wandb", False)
        if self.use_wandb:
            import wandb
            os.environ.setdefault("WANDB_MODE", "offline")
            wandb.init(
                project=getattr(config, "wandb_project", "biai"),
                name=getattr(config, "wandb_run", "bert4-pretrain"),
                config={
                    "learning_rate": config.lr,
                    "batch_size": config.batch_size,
                    "epochs": config.epochs,
                    "device": str(self.device),
                    "loss_weights": self.loss_weights,
                    "tasks": self.enabled_tasks,
                },
            )
            self.wandb = wandb
        else:
            os.environ["WANDB_MODE"] = "disabled"
            self.wandb = None

        self.collates = {
            "mlm": MLMCollateFn(tokenizer, config.seq_len,
                                max_samples=getattr(config, "max_samples", 40),
                                train=True),
            "anp": ANPCollateFn(tokenizer, config.seq_len,
                                max_samples=getattr(config, "max_samples", 40),
                                train=True),
            "big": BIGCollateFn(tokenizer, config.seq_len,
                                max_samples=getattr(config, "max_samples", 40),
                                train=True),
            "gc": GCCollateFn(tokenizer, config.seq_len,
                              max_samples=getattr(config, "max_samples", 40),
                              train=True),
        }
        self.combined_collate = CombinedCollateFn(
            mlm_collate=self.collates["mlm"],
            anp_collate=self.collates["anp"],
            big_collate=self.collates["big"],
            gc_collate=self.collates["gc"],
        )
        self.best_loss = float("inf")
        self.model_save_path = config.checkpoint_save_path
        self.best_path = os.path.join(config.checkpoint_save_path, "bert-best")
        os.makedirs(config.checkpoint_save_path, exist_ok=True)

    # ------------------------------------------------------------------ #
    def train(self, train_dataset, val_dataset=None):
        if getattr(self.config, "use_scheduler", False):
            total_steps = math.ceil(
                len(train_dataset) * self.config.train_sample_ratio
                / self.config.batch_size
            ) * self.config.epochs
            self.scheduler = get_linear_schedule_with_warmup(
                self.optim,
                num_warmup_steps=int(0.3 * total_steps),
                num_training_steps=total_steps,
            )
        for epoch in range(self.config.epochs):
            torch.manual_seed(self.config.seed + epoch)
            np.random.seed(self.config.seed + epoch)
            random.seed(self.config.seed + epoch)

            train_loss = self._run_epoch(train_dataset, epoch, train=True)
            print(f"Epoch {epoch + 1}/{self.config.epochs} train loss: {train_loss:.4f}")

            if val_dataset is not None:
                val_loss = self._run_epoch(val_dataset, epoch, train=False)
                print(f"Epoch {epoch + 1}/{self.config.epochs} val   loss: {val_loss:.4f}")
                if self.wandb is not None:
                    self.wandb.log({"epoch": epoch + 1, "train_loss": train_loss,
                                    "val_loss": val_loss})
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.model.save_pretrained(self.best_path)
                    print(f"Saved best model to {self.best_path}")
                # Snapshot every 5 epochs so intermediate checkpoints survive
                # later training (ablation-friendly).
                if (epoch + 1) % 5 == 0:
                    snap = os.path.join(self.model_save_path, f"bert-epoch-{epoch + 1}")
                    self.model.save_pretrained(snap)
                    print(f"Saved epoch snapshot to {snap}")
            else:
                if self.wandb is not None:
                    self.wandb.log({"epoch": epoch + 1, "train_loss": train_loss})

        if self.wandb is not None:
            self.wandb.finish()
        # Completion marker so external tooling (e.g. pipeline.py) can tell a
        # fully-finished run from a crashed one.
        with open(os.path.join(self.model_save_path, "train-done.json"), "w") as f:
            json.dump({"epochs": self.config.epochs, "best_loss": self.best_loss}, f)

    def _run_epoch(self, dataset, epoch, train=True):
        # Use a fixed validation sample/mask set so best-checkpoint selection
        # compares like with like. Training remains epoch-dependent.
        epoch_seed = self.config.seed + epoch if train else self.config.seed + 100_000
        torch.manual_seed(epoch_seed)
        np.random.seed(epoch_seed)
        random.seed(epoch_seed)
        if train:
            self.model.train()
        else:
            self.model.eval()
        loaders = self._make_loader(dataset, train)
        total_loss, total_batches = 0.0, 0
        desc = "Train" if train else "Valid"
        for loader in loaders:
            for batch in loader:
                if batch["input_ids"].size(0) == 0:
                    continue
                loss = self._train_step(batch) if train else self._eval_step(batch)
                total_loss += loss
                total_batches += 1
        if self.scheduler is not None and train:
            self.scheduler.step()
        return total_loss / max(total_batches, 1)

    def _make_loader(self, dataset, train=True):
        ratio = (self.config.train_sample_ratio if train
                 else getattr(self.config, "val_sample_ratio", 1.0))
        if ratio < 1.0:
            n = max(1, int(len(dataset) * ratio))
            indices = random.sample(range(len(dataset)), n)
            dataset = Subset(dataset, indices)
        task_datasets = [TaskDataset(dataset, t) for t in self.enabled_tasks]
        loaders = []
        for td in task_datasets:
            loaders.append(DataLoader(
                td,
                batch_size=self.config.batch_size,
                shuffle=train,
                num_workers=getattr(self.config, "num_workers", 0),
                collate_fn=self.combined_collate,
                drop_last=False,
            ))
        return loaders

    def _task_step(self, task_name, batch, train):
        """One forward/backward for a single task batch."""
        input_ids = batch["input_ids"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        if task_name == "mlm":
            kwargs["mlm_labels"] = labels
        elif task_name == "anp":
            kwargs["anp_labels"] = labels
        elif task_name == "big":
            kwargs["big_labels"] = labels
        elif task_name == "gc":
            kwargs["gc_labels"] = labels

        if self.use_amp and train:
            with torch.amp.autocast("cuda"):
                out = self.model(**kwargs)
            loss = out["losses"][task_name] * self.loss_weights[task_name]
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            with torch.set_grad_enabled(train):
                out = self.model(**kwargs)
            loss = out["losses"][task_name] * self.loss_weights[task_name]
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
        return loss.item()

    def _train_step(self, batch):
        return self._task_step(batch["task_type"], batch, train=True)

    def _eval_step(self, batch):
        with torch.no_grad():
            return self._task_step(batch["task_type"], batch, train=False)


class BERTFinetuneTrainer:
    """Siamese fine-tuning of CFGFusionModel (CosineEmbeddingLoss)."""

    def __init__(self, model, config):
        self.config = config
        self.device = _resolve_device(config.device)
        self.model = model.to(self.device)
        self.criterion = nn.CosineEmbeddingLoss(margin=getattr(config, "margin", 0.0))

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
        self.use_amp = getattr(config, "use_amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        # Gradient accumulation: effective batch = batch_size * grad_accum
        # (keeps the paper's batch=10 while fitting in GPU memory).
        self.grad_accum = getattr(config, "grad_accum", 1)
        # Let cuDNN cache algorithms for the native graph sizes it encounters.
        torch.backends.cudnn.benchmark = True
        self.best_score = (-1.0, -1.0)
        anchor_pool = getattr(config, "val_anchor_pool", None)
        candidate_pool = getattr(config, "val_candidate_pool", None)
        self.val_anchor_keys = (load_function_keys(anchor_pool)
                                if anchor_pool else None)
        self.val_candidate_keys = (load_function_keys(candidate_pool)
                                   if candidate_pool else None)
        self.selection_metric = (
            "full_pool_mrr10"
            if self.val_anchor_keys is not None
            and self.val_candidate_keys is not None
            else "pair_accuracy"
        )
        os.makedirs(config.checkpoint_save_path, exist_ok=True)

    def _step_optimizer(self, gradient_scale=1.0):
        """Apply accumulated gradients and clear them for the next step."""
        if gradient_scale != 1.0:
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_scale)
        if self.use_amp:
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()
        self.optim.zero_grad(set_to_none=True)

    def train(self, train_dataset, val_dataset=None):
        for epoch in range(self.config.epochs):
            torch.manual_seed(self.config.seed + epoch)
            np.random.seed(self.config.seed + epoch)
            random.seed(self.config.seed + epoch)

            train_loss, train_acc = self._run_epoch(train_dataset, epoch, train=True)
            print(f"Epoch {epoch + 1}/{self.config.epochs} "
                  f"train loss: {train_loss:.4f} acc: {train_acc:.4f}")
            if val_dataset is not None:
                val_loss, val_acc = self._run_epoch(val_dataset, epoch, train=False)
                print(f"Epoch {epoch + 1}/{self.config.epochs} "
                      f"val   loss: {val_loss:.4f} acc: {val_acc:.4f}")
                if (self.val_anchor_keys is not None
                        and self.val_candidate_keys is not None):
                    retrieval = evaluate_retrieval(
                        self.model, val_dataset, self.device,
                        self.val_anchor_keys, self.val_candidate_keys,
                        opt_level=getattr(self.config, "task1_opt", "").upper(),
                    )
                    score = (retrieval["mrr10"], retrieval["rank1"])
                    print(f"Epoch {epoch + 1}/{self.config.epochs} "
                          f"val retrieval MRR10: {score[0]:.4f} "
                          f"Rank1: {score[1]:.4f} "
                          f"({retrieval['anchors']}x{retrieval['candidates']})")
                else:
                    # Kept for small synthetic smoke tests. The real Task 1
                    # entry point always supplies full validation pools.
                    score = (val_acc, 0.0)
                if score > self.best_score:
                    self.best_score = score
                    path = os.path.join(self.config.checkpoint_save_path,
                                        "CFGFusion-best.pth")
                    torch.save(self.model.state_dict(), path)
                    print(f"Saved best model to {path}")
        # Completion marker (crashed runs must not be mistaken for done).
        with open(os.path.join(self.config.checkpoint_save_path, "train-done.json"), "w") as f:
            json.dump({
                "epochs": self.config.epochs,
                "selection_metric": self.selection_metric,
                "best_mrr10": self.best_score[0],
                "best_rank1": self.best_score[1],
            }, f)

    def _pack_by_budget(self, a_ids, a_adj, t_ids, t_adj, labels, budget=900):
        """Split a logical batch by total native nodes without changing loss."""
        groups = []
        current = ([], [], [], [], [])
        current_nodes = 0
        for i in range(len(a_ids)):
            pair_nodes = a_ids[i].shape[0] + t_ids[i].shape[0]
            if current[0] and current_nodes + pair_nodes > budget:
                groups.append(tuple(current))
                current = ([], [], [], [], [])
                current_nodes = 0
            current[0].append(a_ids[i])
            current[1].append(a_adj[i])
            current[2].append(t_ids[i])
            current[3].append(t_adj[i])
            current[4].append(labels[i])
            current_nodes += pair_nodes
            if pair_nodes >= budget:
                groups.append(tuple(current))
                current = ([], [], [], [], [])
                current_nodes = 0
        if current[0]:
            groups.append(tuple(current))
        return groups

    def _run_epoch(self, dataset, epoch, train=True):
        nw = getattr(self.config, "num_workers", 0)
        loader_kwargs = {"num_workers": nw}
        if nw > 0:
            loader_kwargs["prefetch_factor"] = getattr(self.config, "prefetch_factor", 2)
        if train and getattr(self.config, "use_bucketing", True) and hasattr(dataset, "graph_sizes"):
            sizes = dataset.graph_sizes()
            batch_sampler = BucketBatchSampler(sizes, self.config.batch_size,
                                               shuffle=True, seed=self.config.seed + epoch)
            loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=sparse_pair_collate_fn,
                **loader_kwargs,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=train,
                collate_fn=sparse_pair_collate_fn,
                drop_last=False,
                **loader_kwargs,
            )
        total_loss, correct, total = 0.0, 0, 0
        accum = self.grad_accum if train else 1
        if train:
            self.model.train()
            self.optim.zero_grad(set_to_none=True)
        else:
            self.model.eval()
        n_backward = 0
        budget = getattr(self.config, "node_budget", 900)
        for step, batch in enumerate(loader):
            a_ids, a_adj, t_ids, t_adj, labels = batch
            if len(labels) == 0:
                continue
            # Variable-size graphs: lists of per-graph tensors (ids) + sparse
            # COO adjacencies; the model handles native sizes directly.
            a_ids = [ids.to(self.device) for ids in a_ids]
            t_ids = [ids.to(self.device) for ids in t_ids]
            labels = labels.to(self.device)

            # Node-budget packing: large graphs (hundreds of nodes) blow up
            # memory when several are batched together, so a batch is split
            # into groups whose combined node count stays under `budget`
            # (small graphs stay batched; a single oversized graph forms its
            # own group). Each group is forward + backward'd immediately so
            # its activations are freed before the next group — otherwise
            # several large graphs would be alive at once (the backward of
            # the joint loss keeps every group's activations).
            groups = self._pack_by_budget(a_ids, a_adj, t_ids, t_adj, labels,
                                          budget)
            n_total = labels.size(0)
            batch_loss = 0.0
            batch_cosines = []
            for a_g, a_adj_g, t_g, t_adj_g, lab_g in groups:
                lab_g = torch.stack(lab_g).to(self.device)
                n_g = lab_g.size(0)
                with torch.set_grad_enabled(train):
                    with torch.amp.autocast(
                            self.device.type, enabled=self.use_amp):
                        a_emb_g = self.model(a_g, a_adj_g)
                        t_emb_g = self.model(t_g, t_adj_g)
                        raw_loss = self.criterion(
                            a_emb_g, t_emb_g, lab_g.float())
                        backward_loss = raw_loss * (n_g / n_total) / accum
                if train:
                    if self.use_amp:
                        self.scaler.scale(backward_loss).backward()
                    else:
                        backward_loss.backward()
                batch_loss += raw_loss.detach().item() * n_g / n_total
                batch_cosines.append(F.cosine_similarity(
                    a_emb_g.detach(), t_emb_g.detach(), dim=1))
            n_backward += 1
            if train and n_backward % accum == 0:
                self._step_optimizer()
            total_loss += batch_loss * n_total
            with torch.no_grad():
                cos = torch.cat(batch_cosines, dim=0)
                pred = torch.where(cos > 0, torch.tensor(1.0, device=cos.device),
                                   torch.tensor(-1.0, device=cos.device))
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        # Flush the gradient-accumulation tail (last partial group of micro-batches).
        tail = n_backward % accum
        if train and tail:
            # Each tail loss was divided by `accum`; restore the mean over the
            # actual number of micro-batches before stepping.
            self._step_optimizer(gradient_scale=accum / tail)
        return total_loss / max(total, 1), correct / max(total, 1)
