import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from scipy.sparse import coo_matrix
from models.bert import BERT2
from tqdm import tqdm
from utils.utility import tokenize_and_pad
import wandb
from models.model import SemanticAwareModel, SiameseNetwork
from models.tokenizer import AsmTokenizer

# Configuration parameters
class Config:
    batch_size = 5
    max_blocks = 50  # Maximum number of basic blocks
    seq_len = 128    # Maximum sequence length
    hidden_dim = 64  # Graph embedding dimension
    lr = 1e-4
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_checkpoint = vocab_file=os.path.join(".", "outputs", "epoch", "best-model.pth")  # Pretrained BERT path
    save_path = os.path.join(".", "outputs", "semantic_model.pth")  # Model save path

config = Config()

# Custom dataset class
class FunctionPairDataset(Dataset):
    def __init__(self, csv_path, jsonl_path, mapping_path, tokenizer:AsmTokenizer):
        self.df = pd.read_csv(csv_path)
        self.dataset = load_dataset(
            "json", 
            data_files=jsonl_path,
            split="train",
            cache_dir= os.path.join(".", "outputs", "cache"),
            keep_in_memory=False
        )
        with open(mapping_path, "rb") as f:
            self.mapping = pickle.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        a_key = (row["anchor_function_name"], row["anchor_compiler"], 
                str(row["anchor_version"]), row["anchor_opt"], row["anchor_function_file"])
        t_key = (row["target_function_name"], row["target_compiler"], 
                str(row["target_version"]), row["target_opt"], row["target_function_file"])
        
        a_idx = self.mapping[a_key]
        t_idx = self.mapping[t_key]
        
        a_data = self.dataset[a_idx]
        t_data = self.dataset[t_idx]
        
        # Process the first function
        a_input_ids, a_adj = self.process_function(a_data)
        # Process the second function
        t_input_ids, t_adj = self.process_function(t_data)
        
        label = torch.tensor(row["label"], dtype=torch.float32)
        return a_input_ids, a_adj, t_input_ids, t_adj, label

    def process_function(self, func_data):
        # 1. Process instruction blocks
        instr_blocks = func_data["instruction_blocks"]
        processed_blocks = []
        for block in instr_blocks:
            # Tokenize and pad each block
            processed_block = tokenize_and_pad(
                text=block, 
                tokenizer=self.tokenizer, 
                seq_len=config.seq_len  # Reserve space for <CLS> and <SEP>
            )
            processed_blocks.append(torch.tensor(processed_block))
        if len(processed_blocks) >= config.max_blocks:
            processed_blocks = processed_blocks[:config.max_blocks]
        else:
            # Pad with <PAD> tokens if fewer than max_blocks
            # ensure the number of blocks is equal to max_blocks
            for _ in range(config.max_blocks - len(processed_blocks)):
                processed_blocks.append(torch.tensor([self.tokenizer.pad_token_id] * config.seq_len))
        input_ids = torch.stack(processed_blocks, dim=0)
        
        # 2. Process adjacency matrix
        adj_data = func_data["adjacency_matrix"]
        adj = coo_matrix(
            (adj_data["data"], (adj_data["row"], adj_data["col"])),
            shape=(adj_data["shape"])
        ).toarray()

        
        # Adjust adjacency matrix size
        if adj.shape[0] > config.max_blocks:
            adj = adj[:config.max_blocks, :config.max_blocks]
        elif adj.shape[0] < config.max_blocks:
            padded_adj = np.zeros((config.max_blocks, config.max_blocks))
            padded_adj[:adj.shape[0], :adj.shape[1]] = adj
            adj = padded_adj
        
        return input_ids, torch.tensor(adj, dtype=torch.float32)

# Initialize model
def init_model(vocab_size):
    # Load pretrained BERT
    bert_model = BERT2(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=12,
        heads=8,
        seq_len=config.seq_len,
        device=config.device
    )
    bert_model.load_state_dict(torch.load(config.bert_checkpoint))
    
    # Create semantic-aware model
    semantic_model = SemanticAwareModel(
        bert_model=bert_model,
        d_model=128,
        hidden_dim=config.hidden_dim,
        device=config.device
    ).to(config.device)
    
    return SiameseNetwork(semantic_model, config.hidden_dim).to(config.device)

# Update training function to use AMP
def train(model, dataloader, optimizer, criterion, scaler=None, log_freq = 10):  # Add scaler argument
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc="Training")
    for i, (a_ids, a_adj, t_ids, t_adj, labels) in enumerate(progress):
        a_ids, a_adj = a_ids.to(config.device), a_adj.to(config.device)
        t_ids, t_adj = t_ids.to(config.device), t_adj.to(config.device)
        labels = labels.to(config.device)
        
        # Enable autocast for forward pass

        outputs = model(a_ids, a_adj, t_ids, t_adj)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        if scaler is not None:
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular backpropagation
            loss.backward()
            optimizer.step()
        if i % log_freq == 0:
            # Log current batch loss
            wandb.log({"batch_loss": loss.item()})
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    
    return total_loss / len(dataloader)

# Validation function
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for a_ids, a_adj, t_ids, t_adj, labels in dataloader:
            a_ids, a_adj = a_ids.to(config.device), a_adj.to(config.device)
            t_ids, t_adj = t_ids.to(config.device), t_adj.to(config.device)
            labels = labels.to(config.device)
            
            outputs = model(a_ids, a_adj, t_ids, t_adj)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

if __name__ == "__main__":
    os.makedirs(os.path.join(".", "outputs", "epoch"), exist_ok=True)
    # Initialize wandb
    wandb.init(
        project="bert2-training",  # Project name
        config={
            "batch_size": config.batch_size,
            "learning_rate": config.lr,
            "epochs": config.epochs,
            "max_blocks": config.max_blocks,
            "seq_len": config.seq_len,
            "hidden_dim": config.hidden_dim
        }
    )
    wandb.run.name = "bert2-finetuning"  # Set run name
    # Initialize tokenizer to get vocab size
    tokenizer = AsmTokenizer(vocab_file="outputs/baseline-vocab.txt")
    vocab_size = len(tokenizer.vocab)
    
    # Create datasets
    train_dataset = FunctionPairDataset(
        csv_path="outputs/train-function_pool.csv",
        jsonl_path="outputs/baseline-train.jsonl",
        mapping_path="outputs/train-function-idx-mapping.pkl",
        tokenizer=tokenizer
    )
    
    val_dataset = FunctionPairDataset(
        csv_path="outputs/val-function_pool.csv",
        jsonl_path="outputs/baseline-val.jsonl",
        mapping_path="outputs/val-function-idx-mapping.pkl",
        tokenizer=tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = init_model(vocab_size)
    wandb.watch(model, log=None)  # Monitor model parameters
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCELoss()  # Binary cross-entropy
    scaler = torch.amp.GradScaler('cuda')
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',          # Monitor validation loss minimization
        factor=0.5,          # Learning rate decay factor
        patience=3,          # Reduce learning rate after 3 epochs without improvement
        min_lr=1e-6          # Minimum learning rate
    )
    
    best_accuracy = 0
    best_val_loss = float('inf')
    print(f"Starting training on {config.device}...")
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Compute average epoch loss
        train_loss = train(model, train_loader, optimizer, criterion, scaler)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.8f}")
        
        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr
        })
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), config.save_path)
            print(f"Saved new best model with accuracy {val_acc:.4f}")
            # Optionally: save best model to wandb
            wandb.save(config.save_path)
        save_path = os.path.join(".", "outputs", "epoch", f"epoch-{epoch}-semantic_model.pth")
        torch.save(model.state_dict(), save_path)
        
        # Save model with lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Optionally save a separate checkpoint, here using the same file
            # Or use torch.save(model.state_dict(), "best_val_loss_model.pth")
    
    print("Training completed!")
    wandb.finish()  # End wandb run