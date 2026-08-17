"""One-step worst-node Task 1 GPU memory test.

Run with the same allocator setting as train.sh:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
      uv run python tests/stress_gpu_batch.py --nodes 1000
"""
import argparse
import json
import os
from pathlib import Path
import sys

# The allocator reads this before CUDA is initialized (and usually before
# torch is imported), so fail loudly if the caller did not match train.sh.
EXPECTED_ALLOCATOR = "expandable_segments:False"
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != EXPECTED_ALLOCATOR:
    raise RuntimeError(
        "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False before launch"
    )

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from models.bert import BERTForPretraining, build_bert_config
from models.model import CFGFusionModel


def gib(value):
    return round(value / 1024 ** 3, 3)


def make_graph(nodes, seq_len, vocab_size, device):
    # All tokens are visible/non-padding: a conservative BERT activation case.
    ids = torch.randint(
        5, vocab_size, (nodes, seq_len), dtype=torch.long, device=device)
    adj = torch.zeros(nodes, nodes, dtype=torch.float32, device=device)
    positions = torch.arange(nodes - 1, device=device)
    adj[positions, positions + 1] = 1.0
    return ids, adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=967)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 3
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True  # matches BERTFinetuneTrainer

    config = build_bert_config(
        vocab_size=args.vocab_size, seq_len=args.seq_len, d_model=128,
        n_layers=12, heads=8, ff_dim=256)
    bert = BERTForPretraining(config, num_gc_classes=8)
    model = CFGFusionModel(
        bert, d_model=128, mpnn_readout_dim=64, cnn_out=32,
        hidden_dim=64).to(device).train()
    if args.no_gradient_checkpointing:
        model.bert.bert.gradient_checkpointing_disable()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    anchor_ids, anchor_adj = make_graph(
        args.nodes, args.seq_len, args.vocab_size, device)
    target_ids, target_adj = make_graph(
        args.nodes, args.seq_len, args.vocab_size, device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    optimizer.zero_grad(set_to_none=True)
    try:
        with torch.amp.autocast("cuda"):
            anchor_embedding = model([anchor_ids], [anchor_adj])
            target_embedding = model([target_ids], [target_adj])
            loss = F.cosine_embedding_loss(
                anchor_embedding, target_embedding,
                torch.tensor([-1.0], device=device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as error:
        torch.cuda.synchronize(device)
        print(json.dumps({
            "status": "oom",
            "nodes_per_graph": args.nodes,
            "seq_len": args.seq_len,
            "allocator": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
            "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
            "error": str(error),
        }, indent=2))
        return 2

    free_after, _ = torch.cuda.mem_get_info(device)
    print(json.dumps({
        "status": "ok",
        "device": torch.cuda.get_device_name(device),
        "nodes_per_graph": args.nodes,
        "graphs_in_pair": 2,
        "seq_len": args.seq_len,
        "allocator": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        "gradient_checkpointing": model.bert.bert.is_gradient_checkpointing,
        "loss": float(loss.detach()),
        "total_memory_gib": gib(total_memory),
        "free_before_gib": gib(free_before),
        "free_after_gib": gib(free_after),
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
