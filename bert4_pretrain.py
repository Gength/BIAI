"""BERT4 pre-training with the four Order-Matters tasks (MLM/ANP/BIG/GC),
using a HuggingFace BertModel (small config aligned with the paper:
hidden 128, 12 layers, 8 heads, FFN 256, seq len 128).

Usage:
    uv run python bert4_pretrain.py
"""
import os
import torch

from models.tokenizer import AsmTokenizer
from models.bert import build_bert_config, BERTForPretraining
from models.dataset import FunctionDataset, opt_arch_combinations
from models.checkpoint_utils import backup_existing, clear_completion_marker
from models.trainer import BERTPretrainTrainer


class Config:
    # --- data ---
    outputs_dir = "outputs"
    vocab_file = os.path.join("outputs", "baseline-vocab.txt")
    train_jsonl = os.path.join("outputs", "baseline-train.jsonl")
    train_mapping = os.path.join("outputs", "train-function-idx-mapping.pkl")
    val_jsonl = os.path.join("outputs", "baseline-val.jsonl")
    val_mapping = os.path.join("outputs", "val-function-idx-mapping.pkl")

    # --- model (paper config) ---
    seq_len = 128
    d_model = 128
    n_layers = 12
    heads = 8
    ff_dim = 256

    # --- training ---
    batch_size = 32            # pretraining batch (paper does not specify it)
    epochs = 10
    lr = 1e-4                  # paper: Adam, lr 1e-4
    weight_decay = 0.0         # paper specifies Adam, with no weight decay
    betas = (0.9, 0.999)
    use_scheduler = False      # paper: fixed learning rate
    seed = 42
    train_sample_ratio = 1.0   # paper does not describe subsampling
    val_sample_ratio = 1.0
    max_samples = 16           # per-function block samples per task
    num_workers = 0
    resume = False             # full re-run with the 10-epoch schedule

    # --- tasks ---
    train_mlm = True
    train_anp = True
    train_big = True
    train_gc = True
    loss_weights = {"mlm": 1.0, "anp": 1.0, "big": 1.0, "gc": 1.0}

    # --- misc ---
    device = "cuda"          # falls back to CPU automatically
    use_amp = True
    use_wandb = False
    wandb_project = "biai"
    wandb_run = "bert4-pretrain-hf"
    checkpoint_save_path = os.path.join("outputs", "bert4-pretrain-hf")


if __name__ == "__main__":
    config = Config()

    tokenizer = AsmTokenizer(vocab_file=config.vocab_file)
    print(f"Vocab size: {len(tokenizer.vocab)}")

    train_dataset = FunctionDataset(
        dataset_path=config.train_jsonl,
        function_idx_mapping_path=config.train_mapping,
        tokenizer=tokenizer,
        max_len=config.seq_len,
    )
    val_dataset = FunctionDataset(
        dataset_path=config.val_jsonl,
        function_idx_mapping_path=config.val_mapping,
        tokenizer=tokenizer,
        max_len=config.seq_len,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    bert_config = build_bert_config(
        vocab_size=len(tokenizer.vocab),
        seq_len=config.seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        heads=config.heads,
        ff_dim=config.ff_dim,
    )
    resume_path = os.path.join(config.checkpoint_save_path, "bert-best")
    if config.resume and os.path.exists(os.path.join(resume_path, "pytorch_model.bin")):
        model = BERTForPretraining.from_pretrained(resume_path)
        print(f"Resumed pre-trained model from {resume_path}")
    else:
        backup_existing(resume_path)
        model = BERTForPretraining(
            bert_config,
            num_gc_classes=len(opt_arch_combinations),
            tasks=("mlm", "anp", "big", "gc"),
        )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    clear_completion_marker(os.path.join(
        config.checkpoint_save_path, "train-done.json"))
    trainer = BERTPretrainTrainer(model, tokenizer, config)
    trainer.train(train_dataset, val_dataset)
