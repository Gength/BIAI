import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from models.bert import BERT2
from tqdm import tqdm
import wandb
from models.model import CFGFusionModel, SimilarityClassifier
from models.tokenizer import AsmTokenizer
from models.dataset import FunctionPairDataset

# Configuration parameters
class Config:
    batch_size = 25  # RTX 4080: 7, A40: 25
    max_blocks = 50  # Maximum number of basic blocks
    seq_len = 128    # Maximum sequence length
    hidden_dim = 64  # Graph embedding dimension
    lr = 1e-4
    epochs = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_checkpoint = os.path.join(".", "outputs", "bert-pretrain-epoch-8", "bert2"
    "-best.pth")  # Pretrained BERT path
    checkpoint_save_path = os.path.join(".", "outputs", "bert-finetune-epoch")  # Checkpoint save path
    use_amp = True  # Use Automatic Mixed Precision (AMP) if available
    wandb_run = "bert2-finetune"  # Weights & Biases run name
    wandb_project = "bert2-training"  # Weights & Biases project name
config = Config()


class BERT2FinetuneTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        lr=1e-4,
        log_freq=10,
        num_epochs=10,
        model_save_path="",
        device="cuda",
        use_amp=True,
    ):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_freq = log_freq
        self.num_epochs = num_epochs
        self.model_save_path = model_save_path
        self.use_amp = use_amp
        self.best_accuracy = 0
        self.best_val_loss = float('inf')
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        # Mixed precision training
        if use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Initialize wandb
        wandb.init(
            project=config.wandb_project,
            config={
                "batch_size": train_loader.batch_size,
                "learning_rate": lr,
                "epochs": num_epochs,
                "device": device
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
            
            # Training phase
            train_loss = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.scheduler.get_last_lr()[0]
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
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                model_save_path = os.path.join(self.model_save_path, "CFGFusion-best.pth")
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Saved new best model with accuracy {val_acc:.4f}")
            
            # Save checkpoint
            checkpoint_save_path = os.path.join(self.checkpoint_save_path, f"CFGFusion-epoch-{epoch}.pth")
            torch.save(self.model.state_dict(), checkpoint_save_path)
        
        print("Training completed!")
        wandb.finish()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1} Train",
            total=len(self.train_loader),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        
        for i, (a_ids, a_adj, t_ids, t_adj, labels) in enumerate(progress):
            a_ids, a_adj = a_ids.to(self.device), a_adj.to(self.device)
            t_ids, t_adj = t_ids.to(self.device), t_adj.to(self.device)
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
                wandb.log({"batch_train_loss": loss.item()})
        
        return total_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        progress = tqdm(
            self.val_loader, 
            desc=f"Epoch {epoch+1} Valid",
            total=len(self.val_loader),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        
        with torch.no_grad():
            for i, (a_ids, a_adj, t_ids, t_adj, labels) in enumerate(progress):
                a_ids, a_adj = a_ids.to(self.device), a_adj.to(self.device)
                t_ids, t_adj = t_ids.to(self.device), t_adj.to(self.device)
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
                
                if i % self.log_freq == 0:
                    wandb.log({
                        "batch_val_loss": loss.item(),
                        "batch_val_accuracy": accuracy
                    })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        print(f"Val Loss: {avg_loss:.4f}, Val Acc: {accuracy:.4f}")
        return avg_loss, accuracy

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
        csv_path=os.path.join(".", "outputs", "train-function_pool.csv"),
        jsonl_path=os.path.join(".", "outputs", "baseline-train.jsonl"),
        mapping_path=os.path.join(".", "outputs", "train-function-idx-mapping.pkl"),
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        max_blocks=config.max_blocks
    )
    
    val_dataset = FunctionPairDataset(
        csv_path=os.path.join(".", "outputs", "val-function_pool.csv"),
        jsonl_path=os.path.join(".", "outputs", "baseline-val.jsonl"),
        mapping_path=os.path.join(".", "outputs", "val-function-idx-mapping.pkl"),
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        max_blocks=config.max_blocks
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True
    )
    
    # Initialize model and trainer
    model = init_model(vocab_size)
    trainer = BERT2FinetuneTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config.lr,
        num_epochs=config.epochs,
        model_save_path=config.checkpoint_save_path,
        device=config.device,
        use_amp=config.use_amp
    )
    trainer.train()