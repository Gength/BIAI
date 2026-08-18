"""High-throughput Task 1 trainer.

The paper-aligned loss/optimizer/checkpoint logic lives in ``models.trainer``;
this subclass only optimises data movement and graph execution.
"""
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.batch_sampler import WorkloadBatchSampler
from models.collatefn import sparse_pair_collate_fn
from models.trainer import BERTFinetuneTrainer as _BaseTrainer


class BERTFinetuneTrainer(_BaseTrainer):
    """Task 1 trainer that keeps workers alive and batches siamese BERT work."""

    def __init__(self, model, config):
        super().__init__(model, config)
        self._loader_cache = {}

    def _loader(self, dataset, epoch, train):
        key = (id(dataset), train)
        loader = self._loader_cache.get(key)
        if loader is not None:
            if hasattr(loader.batch_sampler, "set_epoch"):
                loader.batch_sampler.set_epoch(epoch)
            return loader

        workers = getattr(self.config, "num_workers", 0)
        kwargs = {
            "num_workers": workers,
            "collate_fn": sparse_pair_collate_fn,
            "pin_memory": self.device.type == "cuda",
        }
        if workers > 0:
            kwargs["prefetch_factor"] = getattr(
                self.config, "prefetch_factor", 2)
            kwargs["persistent_workers"] = True

        if train and getattr(self.config, "use_bucketing", True) \
                and hasattr(dataset, "graph_sizes"):
            loader = DataLoader(
                dataset,
                batch_sampler=WorkloadBatchSampler(
                    dataset.graph_sizes(), self.config.batch_size,
                    shuffle=True, seed=self.config.seed,
                    shapes=(dataset.graph_shapes()
                            if hasattr(dataset, "graph_shapes") else None)),
                **kwargs,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=train,
                drop_last=False,
                **kwargs,
            )
        self._loader_cache[key] = loader
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
        return loader

    def _run_epoch(self, dataset, epoch, train=True):
        loader = self._loader(dataset, epoch, train)
        total_loss, correct, total = 0.0, 0, 0
        accum = self.grad_accum if train else 1
        if train:
            self.model.train()
            self.optim.zero_grad(set_to_none=True)
        else:
            self.model.eval()
        n_backward = 0
        budget = getattr(self.config, "node_budget", 900)

        data_seconds = compute_seconds = 0.0
        window_data = window_compute = 0.0
        window_pairs = window_nodes = window_groups = window_batches = 0
        timing_interval = getattr(self.config, "timing_interval", 500)
        iterator = iter(loader)
        step = 0
        while True:
            wait_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            waited = time.perf_counter() - wait_started
            data_seconds += waited
            window_data += waited
            compute_started = time.perf_counter()
            a_ids, a_adj, t_ids, t_adj, labels = batch
            if len(labels) == 0:
                continue
            batch_nodes = sum(ids.shape[0] for ids in a_ids) + sum(
                ids.shape[0] for ids in t_ids)
            a_ids = [ids.to(self.device, non_blocking=True) for ids in a_ids]
            t_ids = [ids.to(self.device, non_blocking=True) for ids in t_ids]
            labels = labels.to(self.device, non_blocking=True)
            groups = self._pack_by_budget(
                a_ids, a_adj, t_ids, t_adj, labels, budget)
            n_total = labels.size(0)
            batch_loss = 0.0
            batch_cosines = []

            for a_g, a_adj_g, t_g, t_adj_g, lab_g in groups:
                lab_g = torch.stack(lab_g).to(self.device, non_blocking=True)
                n_g = lab_g.size(0)
                with torch.set_grad_enabled(train):
                    with torch.amp.autocast(
                            self.device.type, enabled=self.use_amp):
                        # A single BERT launch now covers anchor and target
                        # blocks, bounded by the same total-node budget.
                        embeddings = self.model(
                            a_g + t_g, a_adj_g + t_adj_g)
                        a_emb_g = embeddings[:n_g]
                        t_emb_g = embeddings[n_g:]
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
                cosine = torch.cat(batch_cosines, dim=0)
                prediction = torch.where(cosine > 0, 1.0, -1.0)
                correct += (prediction == labels).sum().item()
                total += labels.size(0)

            computed = time.perf_counter() - compute_started
            compute_seconds += computed
            window_compute += computed
            window_pairs += n_total
            window_nodes += batch_nodes
            window_groups += len(groups)
            window_batches += 1
            step += 1
            if timing_interval and step % timing_interval == 0:
                self._print_timing(
                    epoch, train, step, window_data, window_compute,
                    window_pairs, window_nodes, window_groups, window_batches)
                window_data = window_compute = 0.0
                window_pairs = window_nodes = window_groups = window_batches = 0

        tail = n_backward % accum
        if train and tail:
            self._step_optimizer(gradient_scale=accum / tail)
        if window_pairs:
            self._print_timing(
                epoch, train, step, window_data, window_compute,
                window_pairs, window_nodes, window_groups, window_batches)
        elapsed = data_seconds + compute_seconds
        if elapsed:
            print(
                f"[throughput] {'train' if train else 'valid'} epoch={epoch + 1} "
                f"loader_wait={100.0 * data_seconds / elapsed:.1f}% "
                f"pairs/s={total / elapsed:.2f}",
                flush=True,
            )
        return total_loss / max(total, 1), correct / max(total, 1)

    @staticmethod
    def _print_timing(epoch, train, step, data_seconds, compute_seconds,
                      pairs, nodes, groups, batches):
        elapsed = data_seconds + compute_seconds
        if not elapsed:
            return
        phase = "train" if train else "valid"
        print(
            f"[throughput] {phase} epoch={epoch + 1} step={step} "
            f"loader_wait={100.0 * data_seconds / elapsed:.1f}% "
            f"pairs/s={pairs / elapsed:.2f} nodes/s={nodes / elapsed:.0f} "
            f"groups/batch={groups / max(batches, 1):.2f}",
            flush=True,
        )
