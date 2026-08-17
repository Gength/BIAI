"""One-step maximum-width BIG pretraining batch GPU memory test."""
import json
import os
from pathlib import Path
import sys

EXPECTED_ALLOCATOR = "expandable_segments:False"
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != EXPECTED_ALLOCATOR:
    raise RuntimeError(
        "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False before launch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from models.bert import BERTForPretraining, build_bert_config


def gib(value):
    return round(value / 1024 ** 3, 3)


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 3
    device = torch.device("cuda")
    vocab_size = 967
    seq_len = 128
    # BIG: batch_size=32, max_samples=16 positives/function plus an equal
    # number of negatives => at most 32 * 16 * 2 = 1024 pair sequences.
    samples = 1024
    config = build_bert_config(
        vocab_size=vocab_size, seq_len=seq_len, d_model=128,
        n_layers=12, heads=8, ff_dim=256)
    model = BERTForPretraining(config, num_gc_classes=8).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    input_ids = torch.randint(
        5, vocab_size, (samples, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    token_type_ids[:, seq_len // 2:] = 1
    labels = torch.randint(0, 2, (samples,), device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    optimizer.zero_grad(set_to_none=True)
    try:
        with torch.amp.autocast("cuda"):
            output = model(
                input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids, big_labels=labels)
            loss = output["losses"]["big"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as error:
        torch.cuda.synchronize(device)
        print(json.dumps({
            "status": "oom",
            "task": "BIG",
            "sequences": samples,
            "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
            "error": str(error),
        }, indent=2))
        return 2

    free_after, _ = torch.cuda.mem_get_info(device)
    print(json.dumps({
        "status": "ok",
        "device": torch.cuda.get_device_name(device),
        "task": "BIG",
        "sequences": samples,
        "seq_len": seq_len,
        "allocator": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        "gradient_checkpointing": model.bert.is_gradient_checkpointing,
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
