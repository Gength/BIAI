import torch
import os
from models.bert import BERT4
from models.model import CFGFusionModel, SimilarityClassifier
from models.tokenizer import AsmTokenizer
from models.dataset import FunctionPairDataset
from models.trainer import BERTFinetuneTrainer
from models.dataset import opt_arch_combinations

# Configuration parameters with sampling ratios
class Config:
    batch_size = 6  # A40: 7
    max_nodes = 200  # Maximum number of basic blocks
    seq_len = 128    # Maximum sequence length
    hidden_dim = 64  # Graph embedding dimension
    lr = 1e-4
    betas = (0.9, 0.999)
    weight_decay = 0.01
    epochs = 12
    device = "cuda"
    bert_checkpoint = os.path.join("outputs", "bert-pretrain", "bert4"
    "-best.pth")  # Pretrained BERT path
    checkpoint_save_path = os.path.join("outputs", "bert-finetune")  # Checkpoint save path
    use_amp = True  # Use Automatic Mixed Precision (AMP) if available
    use_wandb = True  # Use Weights & Biases for logging
    wandb_run = "bert2-finetune"  # Weights & Biases run name
    wandb_project = "bert2-training"  # Weights & Biases project name
    log_freq = 10  
    train_sample_ratio = 0.2  # 20% training set sampling ratio
    val_sample_ratio = 0.2    # 20% validation set sampling ratio

if __name__ == "__main__":
    def init_model(vocab_size):
        # Load pretrained BERT
        bert_model = BERT4(
            vocab_size=vocab_size,
            num_classes=len(opt_arch_combinations),
            d_model=128,
            n_layers=12,
            heads=8,
            seq_len=config.seq_len,
            device=config.device
        )
        bert_model.load_state_dict(torch.load(config.bert_checkpoint))
        
        # Create semantic-aware model
        cfgfusion_model = CFGFusionModel(
            bert_model=bert_model,
            d_model=128,
            hidden_dim=config.hidden_dim,
            device=config.device
        ).to(config.device)
        return SimilarityClassifier(cfgfusion_model, config.hidden_dim).to(config.device)
    config = Config()
    # Initialize tokenizer to get vocab size
    tokenizer = AsmTokenizer(vocab_file="outputs/baseline-vocab.txt")
    vocab_size = len(tokenizer.vocab)
    
    # Create datasets
    train_dataset = FunctionPairDataset(
        function_pool_path=os.path.join("outputs", "train-function_pool.csv"),
        dataset_path=os.path.join("outputs", "baseline-train.jsonl"),
        function_idx_mapping_path=os.path.join("outputs", "train-function-idx-mapping.pkl"),
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        max_nodes=config.max_nodes,
        train=True
    )
    
    val_dataset = FunctionPairDataset(
        function_pool_path=os.path.join("outputs", "val-function_pool.csv"),
        dataset_path=os.path.join("outputs", "baseline-val.jsonl"),
        function_idx_mapping_path=os.path.join("outputs", "val-function-idx-mapping.pkl"),
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        max_nodes=config.max_nodes,
        train=False
    )
    
    # Initialize model and trainer
    model = init_model(vocab_size)
    trainer = BERTFinetuneTrainer(
        model=model,
        train_dataset=train_dataset,  # Pass full training dataset
        val_dataset=val_dataset,      # Pass full validation dataset
        config=config
    )
    trainer.train()