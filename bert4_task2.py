"""Task 2 (paper): graph classification of the optimization option (O0-O3).

The graph embedding (CFGFusionModel) is classified with softmax + cross
entropy; the two platforms (x64 / arm64) are trained and evaluated
separately, as the paper reports them separately (Table 3).

Usage:
    uv run python bert4_task2.py --platform x64
    uv run python bert4_task2.py --platform arm64
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.batch_sampler import WorkloadBatchSampler
from models.tokenizer import AsmTokenizer
from models.bert import BERTForPretraining
from models.checkpoint_utils import backup_existing, clear_completion_marker
from models.model import CFGFusionModel
from models.graph_dataset import Task2Dataset
from models.trainer import _resolve_device


class Config:
    seq_len = 128
    d_model = 128
    mpnn_readout_dim = 64
    cnn_out = 32
    graph_hidden_dim = 64
    checkpoint_node_threshold = 1536
    num_classes = 4  # O0, O1, O2, O3

    batch_size = 10   # paper: batch size 10
    epochs = 15
    lr = 1e-4
    weight_decay = 0.0         # paper specifies Adam, with no weight decay
    betas = (0.9, 0.999)
    seed = 42
    num_workers = 2            # 2 workers 并行 tokenize（4 个会爆 15GB RAM）；内存 ~7GB 安全
    prefetch_factor = 4
    device = "cuda"
    use_amp = True
    node_budget = 4000      # four worst-case 1000-node CFGs per memory group
    pretrained_path = os.path.join("outputs", "bert4-pretrain-hf", "bert-best")
    checkpoint_save_path = os.path.join("outputs", "bert4-task2-hf")


def collate(batch):
    """batch: list of (input_ids [N,L], adj COO, label) -> variable-size lists."""
    ids_list = [ids for ids, _, _ in batch]
    adj_list = [adj for _, adj, _ in batch]
    labels = torch.tensor([lbl.item() for _, _, lbl in batch], dtype=torch.long)
    return ids_list, adj_list, labels


def _loader_kwargs(config, device):
    """DataLoader settings shared by train/validation/test."""
    workers = config.num_workers
    kwargs = {
        "num_workers": workers,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
        kwargs["persistent_workers"] = True
    return kwargs


def _pack_by_node_budget(ids, adj, labels, budget):
    """Split a logical batch for memory while preserving its mean loss."""
    groups = []
    current = ([], [], [])
    current_nodes = 0
    for graph_ids, graph_adj, label in zip(ids, adj, labels):
        n_nodes = graph_ids.shape[0]
        if current[0] and current_nodes + n_nodes > budget:
            groups.append(tuple(current))
            current = ([], [], [])
            current_nodes = 0
        current[0].append(graph_ids)
        current[1].append(graph_adj)
        current[2].append(label)
        current_nodes += n_nodes
        if n_nodes >= budget:
            groups.append(tuple(current))
            current = ([], [], [])
            current_nodes = 0
    if current[0]:
        groups.append(tuple(current))
    return groups


def run_epoch(model, classifier, loader, device, train, config, optim=None,
              scaler=None):
    if train:
        model.train()
        classifier.train()
    else:
        model.eval()
        classifier.eval()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = bool(config.use_amp and device.type == "cuda")
    for ids, adj, labels in loader:
        ids = [t.to(device, non_blocking=True) for t in ids]
        labels = labels.to(device, non_blocking=True)
        groups = _pack_by_node_budget(
            ids, adj, labels, getattr(config, "node_budget", 900))
        n_batch = labels.size(0)
        if train:
            if optim is None:
                raise ValueError("training requires an optimizer")
            optim.zero_grad(set_to_none=True)
        for ids_g, adj_g, labels_g in groups:
            labels_g = torch.stack(labels_g).to(device)
            with torch.set_grad_enabled(train):
                with torch.amp.autocast(device.type, enabled=use_amp):
                    emb = model(ids_g, adj_g)
                    logits = classifier(emb)
                    loss = nn.functional.cross_entropy(logits, labels_g)
                    backward_loss = loss * (labels_g.size(0) / n_batch)
            if train:
                if use_amp:
                    scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()
            total_loss += loss.item() * labels_g.size(0)
            correct += (logits.argmax(dim=1) == labels_g).sum().item()
            total += labels_g.size(0)
        if train:
            if use_amp:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Task2: optimization-level classification")
    parser.add_argument("--platform", choices=["x64", "arm64"], default="x64")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval", action="store_true",
                        help="evaluate the best checkpoint on the test split")
    args = parser.parse_args()
    config = Config()
    device = _resolve_device(args.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    tokenizer = AsmTokenizer(vocab_file=os.path.join("outputs", "baseline-vocab.txt"))
    print(f"Vocab size: {len(tokenizer.vocab)}")

    def make_dataset(split):
        return Task2Dataset(
            function_list_path=os.path.join(
                "outputs", f"task2-{args.platform}-{split}-functions.pkl"),
            dataset_path=os.path.join("outputs", f"baseline-{split}.jsonl"),
            function_idx_mapping_path=os.path.join(
                "outputs", f"{split}-function-idx-mapping.pkl"),
            tokenizer=tokenizer, seq_len=config.seq_len,
        )

    if args.eval:
        test_ds = make_dataset("test")
        print(f"Task2-{args.platform}: test={len(test_ds)}")
        checkpoint = torch.load(
            os.path.join(config.checkpoint_save_path,
                         f"CFGFusion-task2-{args.platform}-best.pth"),
            map_location=device, weights_only=True)
        bert = BERTForPretraining.from_pretrained(config.pretrained_path)
        model = CFGFusionModel(bert, d_model=config.d_model,
                               mpnn_readout_dim=config.mpnn_readout_dim,
                               cnn_out=config.cnn_out,
                               hidden_dim=config.graph_hidden_dim,
                               checkpoint_node_threshold=(
                                   config.checkpoint_node_threshold)).to(device)
        classifier = nn.Linear(config.graph_hidden_dim, config.num_classes).to(device)
        model.load_state_dict(checkpoint["model"])
        classifier.load_state_dict(checkpoint["classifier"])
        test_loader = DataLoader(test_ds, batch_sampler=WorkloadBatchSampler(
                                    test_ds.graph_sizes(), config.batch_size,
                                    shuffle=False, seed=config.seed,
                                    shapes=test_ds.graph_shapes()),
                                **_loader_kwargs(config, device))
        loss, acc = run_epoch(model, classifier, test_loader, device, False, config)
        print(f"Task2-{args.platform} test: loss {loss:.4f} accuracy {acc:.4f}")
        return

    train_ds = make_dataset("train")
    val_ds = make_dataset("val")
    print(f"Task2-{args.platform}: train={len(train_ds)} val={len(val_ds)}")

    bert = BERTForPretraining.from_pretrained(config.pretrained_path)
    model = CFGFusionModel(bert, d_model=config.d_model,
                           mpnn_readout_dim=config.mpnn_readout_dim,
                           cnn_out=config.cnn_out,
                           hidden_dim=config.graph_hidden_dim,
                           checkpoint_node_threshold=(
                               config.checkpoint_node_threshold)).to(device)
    classifier = nn.Linear(config.graph_hidden_dim, config.num_classes).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in classifier.parameters()):,}")

    backup_existing(os.path.join(
        config.checkpoint_save_path,
        f"CFGFusion-task2-{args.platform}-best.pth"))
    clear_completion_marker(os.path.join(
        config.checkpoint_save_path,
        f"train-done-{args.platform}.json"))

    params = list(model.parameters()) + list(classifier.parameters())
    optim = torch.optim.Adam(params, lr=config.lr, betas=config.betas,
                              weight_decay=config.weight_decay)
    scaler = (torch.amp.GradScaler("cuda")
              if config.use_amp and device.type == "cuda" else None)

    train_loader = DataLoader(train_ds, batch_sampler=WorkloadBatchSampler(
                                  train_ds.graph_sizes(), config.batch_size,
                                  shuffle=True, seed=config.seed,
                                  shapes=train_ds.graph_shapes()),
                              **_loader_kwargs(config, device))
    val_loader = DataLoader(val_ds, batch_sampler=WorkloadBatchSampler(
                                val_ds.graph_sizes(), config.batch_size,
                                shuffle=False, seed=config.seed,
                                shapes=val_ds.graph_shapes()),
                            **_loader_kwargs(config, device))

    best_acc = -1.0
    os.makedirs(config.checkpoint_save_path, exist_ok=True)
    for epoch in range(config.epochs):
        train_loader.batch_sampler.set_epoch(epoch)
        tr_loss, tr_acc = run_epoch(model, classifier, train_loader, device,
                                    True, config, optim, scaler)
        va_loss, va_acc = run_epoch(model, classifier, val_loader, device,
                                    False, config)
        print(f"Epoch {epoch + 1}/{config.epochs} [{args.platform}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            path = os.path.join(config.checkpoint_save_path,
                                f"CFGFusion-task2-{args.platform}-best.pth")
            torch.save({"model": model.state_dict(),
                        "classifier": classifier.state_dict()}, path)
            print(f"Saved best model to {path}")
    print(f"Best val accuracy ({args.platform}): {best_acc:.4f}")
    with open(os.path.join(config.checkpoint_save_path, f"train-done-{args.platform}.json"), "w") as f:
        json.dump({"epochs": config.epochs, "best_acc": best_acc}, f)


if __name__ == "__main__":
    main()
