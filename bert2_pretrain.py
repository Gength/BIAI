import os
from models.bert import BERT2
from models.tokenizer import AsmTokenizer
from models.dataset import FunctionDataset
from models.trainer import BERT2PretrainTrainer
import torch.nn as nn

class Config:
    batch_size = 10
    epochs = 15
    log_freq = 10
    seq_len = 128    # Maximum sequence length
    lr = 1e-5
    weight_decay=0.01
    betas=(0.9, 0.999)
    device = "cuda"
    checkpoint_save_path = os.path.join("outputs", "bert2-improved-pretrain-lrscheduler")
    use_amp = True  # Use Automatic Mixed Precision (AMP) if available
    use_wandb = True  # Use Weights & Biases for logging
    wandb_run = "bert2-improved-pretrain-lrscheduler"  # Weights & Biases run name
    wandb_project = "bert4-training"  # Weights & Biases project name
    train_sample_ratio = 0.2  # 20% training set sampling ratio
    val_sample_ratio = 0.2    # 20% validation set sampling ratio

# Dummy context manager for non-mixed precision training
class dummy_context:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        pass

if __name__ == "__main__":
    config = Config()
    tokenizer = AsmTokenizer(
        vocab_file=os.path.join("outputs", f"baseline-vocab.txt")
    )
    print(f"Vocab size: {len(tokenizer.vocab)}")
    # Set cache directory
    cache_dir = os.path.join("outputs", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Load training set (enable memory mapping)
    train_function_dataset = FunctionDataset(
        dataset_path=os.path.join("outputs", f"baseline-train.jsonl"),
        function_idx_mapping_path=os.path.join("outputs", f"train-function-idx-mapping.pkl"),
        tokenizer=tokenizer,
        max_len=config.seq_len
    )
    print(f"Train Dataset size: {len(train_function_dataset)}")

    # Load validation set (enable memory mapping)
    val_function_dataset = FunctionDataset(
        dataset_path=os.path.join("outputs", f"baseline-val.jsonl"),
        function_idx_mapping_path=os.path.join("outputs", f"val-function-idx-mapping.pkl"),
        tokenizer=tokenizer,
        max_len=config.seq_len
    )
    print(f"Validation Dataset size: {len(val_function_dataset)}")
    
    # Create model
    bert_model = BERT2(
        vocab_size=len(tokenizer.vocab),
        d_model=128,
        n_layers=12,
        heads=8,
        seq_len=config.seq_len,
        device=config.device
    )
    # initialize model weights
    for module in bert_model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    # Create trainers
    trainer = BERT2PretrainTrainer(
        model=bert_model,
        train_dataset=train_function_dataset,   # Pass full training dataset
        valid_dataset=val_function_dataset,   # Pass full validation dataset
        tokenizer=tokenizer,
        config=config
    )
    trainer.train()