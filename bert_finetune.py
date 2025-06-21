import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from models.bert import BERT2
from tqdm import tqdm
import wandb
from models.model import CFGFusionModel, SimilarityClassifier
from models.tokenizer import AsmTokenizer
from models.dataset import FunctionPairDataset
import math

# Configuration parameters with sampling ratios
class Config:
    batch_size = 6  # A40: 7
    max_nodes = 200  # Maximum number of basic blocks
    seq_len = 128    # Maximum sequence length
    hidden_dim = 64  # Graph embedding dimension
    lr = 1e-4
    epochs = 12
    device = "cuda"
    bert_checkpoint = os.path.join("outputs", "bert-pretrain-epoch-8", "bert2"
    "-best.pth")  # Pretrained BERT path
    checkpoint_save_path = os.path.join("outputs", "bert-finetune")  # Checkpoint save path
    use_amp = True  # Use Automatic Mixed Precision (AMP) if available
    use_wandb = True  # Use Weights & Biases for logging
    wandb_run = "bert2-finetune"  # Weights & Biases run name
    wandb_project = "bert2-training"  # Weights & Biases project name
    train_sample_ratio = 0.2  # 20% training set sampling ratio
    val_sample_ratio = 0.2    # 20% validation set sampling ratio

config = Config()


class BERT2FinetuneTrainer:
    def __init__(
        self,
        model,
        train_dataset,  # Full training dataset
        val_dataset,    # Full validation dataset
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        log_freq=10,
        num_epochs=10,
        model_save_path="",
        device="cuda",
        use_amp=True,
    ):
        self.device = device
        self.model = model.to(device)
        self.train_dataset = train_dataset  # Save full training dataset
        self.val_dataset = val_dataset      # Save full validation dataset
        self.log_freq = log_freq
        self.num_epochs = num_epochs
        self.model_save_path = model_save_path
        self.use_amp = use_amp
        self.best_accuracy = 0
        self.best_val_loss = float('inf')
        
        # DataLoader parameters
        self.batch_size = config.batch_size
        self.num_workers = 8
        self.prefetch_factor = 4
        self.persistent_workers = True
        self.pin_memory = True
        
        # Sampling seed
        self.sample_seed = 42
        
        # Initialize sampling state
        self.train_sampled_indices = set()
        self.train_unsampled_indices = set(range(len(train_dataset)))
        self.val_sampled_indices = set()
        self.val_unsampled_indices = set(range(len(val_dataset)))
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Calculate total global steps
        train_samples_per_epoch = int(len(train_dataset) * config.train_sample_ratio)
        steps_per_epoch = math.ceil(train_samples_per_epoch / config.batch_size)
        total_steps = steps_per_epoch * num_epochs

        # Initialize global learning rate scheduler
        self.optim_schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            final_div_factor=1e2,
            div_factor=25,
            three_phase=False
        )
        
        # Mixed precision training
        if use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        if config.use_wandb:
            os.environ["WANDB_MODE"] = "online"  # Enable Weights & Biases logging
        else:
            os.environ["WANDB_MODE"] = "disabled"  # Disable Weights & Biases logging
        # Initialize wandb
        wandb.init(
            project=config.wandb_project,
            config={
                "batch_size": config.batch_size,
                "learning_rate": lr,
                "epochs": num_epochs,
                "device": device,
                "train_sample_ratio": config.train_sample_ratio,
                "val_sample_ratio": config.val_sample_ratio,
                "total_training_steps": total_steps
            }
        )
        wandb.run.name = config.wandb_run
        wandb.watch(self.model, log=None)
        
        # Ensure output directory exists
        os.makedirs(config.checkpoint_save_path, exist_ok=True)

    def train(self):
        print(f"Starting training on {self.device}...")
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Set random seed for reproducibility
            self.sample_seed = epoch + 1  # Use a different seed for each epoch
            torch.manual_seed(self.sample_seed)
            np.random.seed(self.sample_seed)
            random.seed(self.sample_seed)
            
            # Sample training set (use non-repetitive sampling)
            train_indices = self.sample_dataset_indices(
                self.train_dataset, 
                config.train_sample_ratio,
                "training"
            )
            
            # Sample validation set (use non-repetitive sampling)
            val_indices = self.sample_dataset_indices(
                self.val_dataset, 
                config.val_sample_ratio,
                "validation"
            )
            
            # Create sampled datasets
            train_subset = Subset(self.train_dataset, train_indices)
            val_subset = Subset(self.val_dataset, val_indices)
            
            # Create DataLoaders
            train_loader = self.create_data_loader(train_subset, shuffle=True)
            val_loader = self.create_data_loader(val_subset, shuffle=False)
            
            # Training phase
            train_loss = self.train_epoch(epoch, train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(epoch, val_loader)
            
            # Log epoch metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            })
            
            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                model_save_path = os.path.join(self.model_save_path, "CFGFusion-best.pth")
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Saved new best model with accuracy {val_acc:.4f}")
            
            # Save checkpoint
            checkpoint_save_path = os.path.join(self.model_save_path, f"CFGFusion-epoch-{epoch}.pth")
            torch.save(self.model.state_dict(), checkpoint_save_path)
        
        print("Training completed!")
        wandb.finish()

    def sample_dataset_indices(self, dataset, sample_ratio, dataset_type):
        """Non-repetitive sampling, continue sampling after reset"""
        if dataset_type == "training":
            unsampled = self.train_unsampled_indices
            sampled = self.train_sampled_indices
        else:
            unsampled = self.val_unsampled_indices
            sampled = self.val_sampled_indices

        total_size = len(dataset)
        sample_size = int(total_size * sample_ratio)

        # If there are not enough unsampled samples, reset the sampling state
        if len(unsampled) < sample_size:
            print(f"Resetting {dataset_type} sampling pool after {len(sampled)} samples seen")
            # Put the sampled samples back into the unsampled pool
            unsampled.update(sampled)
            sampled.clear()

        # Randomly select samples from the unsampled pool
        selected_indices = random.sample(list(unsampled), sample_size)

        # Update sampled and unsampled sets
        sampled.update(selected_indices)
        unsampled.difference_update(selected_indices)

        print(f"Sampling {sample_size} {dataset_type} samples "
              f"({len(sampled)}/{total_size} total sampled)")
        
        return selected_indices

    def create_data_loader(self, dataset, shuffle=True):
        """Create DataLoader"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory
        )

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        total_loss = 0
        progress = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1} Train",
            total=len(train_loader),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        
        for i, (a_ids, a_adj, t_ids, t_adj, labels) in enumerate(progress):
            a_ids, t_ids = a_ids.to(self.device), t_ids.to(self.device)
            a_adj, t_adj = a_adj.to(self.device), t_adj.to(self.device)
            labels = labels.to(self.device)
            
            with torch.autocast(device_type=self.device, enabled=self.use_amp):
                outputs = self.model(a_ids, a_adj, t_ids, t_adj)
                loss = self.criterion(outputs, labels)
            
            # Backpropagation
            self.optimizer.zero_grad()
            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
            
            if i % self.log_freq == 0:
                wandb.log({
                    "batch_train_loss": loss.item(),
                    "lr": self.optim_schedule.get_last_lr()[0]
                })
            
            # Update learning rate
            self.optim_schedule.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f}")
        return avg_loss

    def validate_epoch(self, epoch, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        progress = tqdm(
            val_loader, 
            desc=f"Epoch {epoch+1} Valid",
            total=len(val_loader),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        
        with torch.no_grad():
            for i, (a_ids, a_adj, t_ids, t_adj, labels) in enumerate(progress):
                a_ids, t_ids = a_ids.to(self.device), t_ids.to(self.device)
                a_adj, t_adj = a_adj.to(self.device), t_adj.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(a_ids, a_adj, t_ids, t_adj)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                correct += (predictions.to(torch.int32) == labels.to(torch.int32)).sum().item()
                total += labels.size(0)
                
                # Logging
                total_loss += loss.item()
                accuracy = correct / total if total > 0 else 0
                progress.set_postfix(loss=loss.item(), accuracy=accuracy)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        print(f"Val Loss: {avg_loss:.4f}, Val Acc: {accuracy:.4f}")
        return avg_loss, accuracy

# Initialize model (unchanged)
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
    cfgfusion_model = CFGFusionModel(
        bert_model=bert_model,
        d_model=128,
        hidden_dim=config.hidden_dim,
        device=config.device
    ).to(config.device)
    
    return SimilarityClassifier(cfgfusion_model, config.hidden_dim).to(config.device)

if __name__ == "__main__":
    # Initialize tokenizer to get vocab size
    tokenizer = AsmTokenizer(vocab_file="outputs/baseline-vocab.txt")
    vocab_size = len(tokenizer.vocab)
    
    # Create datasets
    train_dataset = FunctionPairDataset(
        csv_path=os.path.join("outputs", "train-function_pool.csv"),
        jsonl_path=os.path.join("outputs", "baseline-train.jsonl"),
        mapping_path=os.path.join("outputs", "train-function-idx-mapping.pkl"),
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        max_nodes=config.max_nodes
    )
    
    val_dataset = FunctionPairDataset(
        csv_path=os.path.join("outputs", "val-function_pool.csv"),
        jsonl_path=os.path.join("outputs", "baseline-val.jsonl"),
        mapping_path=os.path.join("outputs", "val-function-idx-mapping.pkl"),
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        max_nodes=config.max_nodes
    )
    
    # Initialize model and trainer
    model = init_model(vocab_size)
    trainer = BERT2FinetuneTrainer(
        model=model,
        train_dataset=train_dataset,  # Pass full training dataset
        val_dataset=val_dataset,      # Pass full validation dataset
        lr=config.lr,
        num_epochs=config.epochs,
        model_save_path=config.checkpoint_save_path,
        device=config.device,
        use_amp=config.use_amp
    )
    trainer.train()