"""A/B benchmark for legacy per-graph vs fused Task 1 BERT execution.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
      uv run python tests/benchmark_gpu_finetune.py
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

EXPECTED_ALLOCATOR = "expandable_segments:False"
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != EXPECTED_ALLOCATOR:
    raise RuntimeError(
        "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False before launch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from models.bert import BERTForPretraining, build_bert_config
from models.model import CFGFusionModel


def make_graphs(count, nodes, seq_len, vocab_size, device):
    graphs, adjacencies = [], []
    for _ in range(count):
        graphs.append(torch.randint(
            5, vocab_size, (nodes, seq_len), dtype=torch.long, device=device))
        adjacency = torch.zeros(nodes, nodes, device=device)
        positions = torch.arange(nodes - 1, device=device)
        adjacency[positions, positions + 1] = 1
        adjacencies.append(adjacency)
    return graphs, adjacencies


def build_model(vocab_size, seq_len, device):
    config = build_bert_config(
        vocab_size=vocab_size, seq_len=seq_len, d_model=128,
        n_layers=12, heads=8, ff_dim=256)
    return CFGFusionModel(
        BERTForPretraining(config, num_gc_classes=8),
        d_model=128, mpnn_readout_dim=64, cnn_out=32,
        hidden_dim=64).to(device).train()


def run(mode, args, device):
    model = build_model(args.vocab_size, args.seq_len, device)
    if mode == "fused_checkpointed":
        model.adaptive_gradient_checkpointing = False
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    anchor_ids, anchor_adj = make_graphs(
        args.graphs, args.nodes, args.seq_len, args.vocab_size, device)
    target_ids, target_adj = make_graphs(
        args.graphs, args.nodes, args.seq_len, args.vocab_size, device)
    labels = torch.ones(args.graphs, device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            if mode == "legacy_per_graph":
                anchors = torch.cat([
                    model._forward_batch(ids.unsqueeze(0), adj.unsqueeze(0))
                    for ids, adj in zip(anchor_ids, anchor_adj)
                ])
                targets = torch.cat([
                    model._forward_batch(ids.unsqueeze(0), adj.unsqueeze(0))
                    for ids, adj in zip(target_ids, target_adj)
                ])
            else:
                embeddings = model(
                    anchor_ids + target_ids, anchor_adj + target_adj)
                anchors = embeddings[:args.graphs]
                targets = embeddings[args.graphs:]
            loss = F.cosine_embedding_loss(anchors, targets, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "mode": mode,
        "seconds": round(elapsed, 4),
        "seconds_per_step": round(elapsed / args.steps, 4),
        "pairs_per_second": round(args.steps * args.graphs / elapsed, 2),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device) / 1024 ** 3, 3),
        "peak_reserved_gib": round(
            torch.cuda.max_memory_reserved(device) / 1024 ** 3, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=967)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 3
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

    results = [run(mode, args, device) for mode in (
        "legacy_per_graph", "fused_checkpointed", "fused_adaptive")]
    legacy_seconds = results[0]["seconds_per_step"]
    for result in results[1:]:
        result["speedup_vs_legacy"] = round(
            legacy_seconds / result["seconds_per_step"], 2)
    results[2]["speedup_vs_fused_checkpointed"] = round(
        results[1]["seconds_per_step"] / results[2]["seconds_per_step"], 2)
    print(json.dumps({
        "device": torch.cuda.get_device_name(device),
        "graphs_per_side": args.graphs,
        "nodes_per_graph": args.nodes,
        "total_nodes_per_pair_batch": 2 * args.graphs * args.nodes,
        "steps": args.steps,
        "results": results,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
