"""Fine-tuning of CFGFusionModel (BERT + MPNN + ResNet11) with a siamese
network and cosine embedding loss, on the cross-platform function
similarity task (paper task 1).

Usage:
    uv run python bert4_finetune.py
"""
import os
import argparse

from models.tokenizer import AsmTokenizer
from models.bert import BERTForPretraining
from models.checkpoint_utils import backup_existing, clear_completion_marker
from models.model import CFGFusionModel
from models.graph_dataset import FunctionPairDataset
from models.finetune_trainer import BERTFinetuneTrainer


class Config:
    # --- data ---
    vocab_file = os.path.join("outputs", "baseline-vocab.txt")
    train_pool = None   # set from --task1_opt
    train_jsonl = os.path.join("outputs", "baseline-train.jsonl")
    train_mapping = os.path.join("outputs", "train-function-idx-mapping.pkl")
    val_pool = None     # set from --task1_opt
    val_jsonl = os.path.join("outputs", "baseline-val.jsonl")
    val_mapping = os.path.join("outputs", "val-function-idx-mapping.pkl")

    # --- model ---
    seq_len = 128
    d_model = 128
    mpnn_readout_dim = 64
    cnn_out = 32
    graph_hidden_dim = 64
    checkpoint_node_threshold = 1536

    # --- training ---
    batch_size = 10           # paper effective batch, now executed as one logical batch
    grad_accum = 1
    epochs = 15
    lr = 1e-4                # paper: Adam, lr 1e-4
    weight_decay = 0.0         # paper specifies Adam, with no weight decay
    betas = (0.9, 0.999)
    margin = 0.0
    seed = 42
    num_workers = 2            # 2 workers 并行 tokenize（4 个会爆 15GB RAM）；内存 ~7GB 安全
    prefetch_factor = 2        # same queued sample count as the old 2x4x5 setup
    node_budget = 4000         # up to two worst-case 1000+1000 pairs per launch
    timing_interval = 200      # report loader wait vs GPU compute during long epochs

    # --- misc ---
    device = "cuda"          # falls back to CPU automatically
    use_amp = True
    pretrained_path = os.path.join("outputs", "bert4-pretrain-hf", "bert-best")
    checkpoint_save_path = os.path.join("outputs", "bert4-finetune-hf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task1 siamese fine-tuning")
    parser.add_argument("--task1_opt", choices=["o2", "o3"], default="o2",
                        help="paper Task 1 dataset: gcc-O2 or gcc-O3")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the Config epochs (for ablations)")
    args = parser.parse_args()

    config = Config()
    if args.epochs is not None:
        config.epochs = args.epochs
    config.train_pool = os.path.join(
        "outputs", f"task1-{args.task1_opt}-train-function_pool.csv")
    config.val_pool = os.path.join(
        "outputs", f"task1-{args.task1_opt}-val-function_pool.csv")
    config.task1_opt = args.task1_opt
    config.val_anchor_pool = os.path.join(
        "outputs", "task2-x64-val-functions.pkl")
    config.val_candidate_pool = os.path.join(
        "outputs", "task2-arm64-val-functions.pkl")
    config.checkpoint_save_path = os.path.join(
        "outputs", f"bert4-finetune-hf-{args.task1_opt}")

    tokenizer = AsmTokenizer(vocab_file=config.vocab_file)
    print(f"Vocab size: {len(tokenizer.vocab)}")

    train_dataset = FunctionPairDataset(
        function_pool_path=config.train_pool,
        dataset_path=config.train_jsonl,
        function_idx_mapping_path=config.train_mapping,
        tokenizer=tokenizer,
        seq_len=config.seq_len,
    )
    val_dataset = FunctionPairDataset(
        function_pool_path=config.val_pool,
        dataset_path=config.val_jsonl,
        function_idx_mapping_path=config.val_mapping,
        tokenizer=tokenizer,
        seq_len=config.seq_len,
    )
    print(f"Train pairs: {len(train_dataset)}, Val pairs: {len(val_dataset)}")

    bert = BERTForPretraining.from_pretrained(config.pretrained_path)
    model = CFGFusionModel(
        bert,
        d_model=config.d_model,
        mpnn_readout_dim=config.mpnn_readout_dim,
        cnn_out=config.cnn_out,
        hidden_dim=config.graph_hidden_dim,
        checkpoint_node_threshold=config.checkpoint_node_threshold,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    backup_existing(os.path.join(
        config.checkpoint_save_path, "CFGFusion-best.pth"))
    clear_completion_marker(os.path.join(
        config.checkpoint_save_path, "train-done.json"))
    trainer = BERTFinetuneTrainer(model, config)
    trainer.train(train_dataset, val_dataset)
