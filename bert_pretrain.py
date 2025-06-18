import argparse
import os
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.bert import BERT2
from models.tokenizer import AsmTokenizer
from datasets import load_dataset
from models.dataset import TaskDataset
from models.collatefn import MLMCollateFn, ANPCollateFn, CombinedCollateFn
import wandb
import time
class Config:
    batch_size = 30  # RTX 4080: 10, A40: 30
    epochs = 8
    seq_len = 128    # Maximum sequence length
    lr = 1e-4
    epochs = 8
    device = "cuda"
    checkpoint_save_path = os.path.join("outputs", "bert-pretrain")
    use_amp = True  # Use Automatic Mixed Precision (AMP) if available
    wandb_run = "bert2-pretrain"  # Weights & Biases run name
    wandb_project = "bert2-training"  # Weights & Biases project name
config = Config()
class BERT2PretrainTrainer:
    def __init__(
        self,
        model,
        mlm_train_loader,
        anp_train_loader,
        mlm_valid_loader,
        anp_valid_loader,
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
        self.mlm_train_loader = mlm_train_loader
        self.anp_train_loader = anp_train_loader
        self.mlm_valid_loader = mlm_valid_loader
        self.anp_valid_loader = anp_valid_loader
        
        # Single optimizer for the whole model
        self.optim = Adam(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        
        # Calculate total number of iterations
        total_steps = max(len(mlm_train_loader), len(anp_train_loader)) * num_epochs
        
        # Single learning rate scheduler
        self.optim_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max=total_steps,
            eta_min=1e-6  # 最低学习率
        )
        # self.optim_schedule = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optim,
        #     max_lr=1e-4,
        #     total_steps=total_steps,
        #     pct_start=0.1,
        #     anneal_strategy="cos",
        #     final_div_factor=1e2,
        # )

        self.log_freq = log_freq
        self.best_loss = float('inf')
        self.model_save_path = model_save_path
        self.num_epochs = num_epochs
        self.use_amp = use_amp
        if use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
        # Initialize wandb
        wandb.init(project="bert2-training", config={
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": mlm_train_loader.batch_size,
            "epochs": num_epochs,
            "device": device
        })
        wandb.watch(self.model, log=None)

        # if os.path.exists(os.path.join(".", "outputs", "epoch")):
        #     file_list = os.listdir(os.path.join(".", "outputs", "epoch"))
        #     for file_name in file_list:
        #         if os.path.isfile(os.path.join(".", "outputs", "epoch", file_name)):
        #             os.remove(os.path.join(".", "outputs", "epoch", file_name))  # Clear previous epoch files
        # else:
        #     os.makedirs(os.path.join(".", "outputs", "epoch"), exist_ok=True)  # Ensure output directory exists
        os.makedirs(self.model_save_path, exist_ok=True)

    def train(self):
        with open(os.path.join(".", "outputs", "bert-pretrain-epoch", "best-model-log.txt"), "w") as f:
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for epoch in range(self.num_epochs):
            # Training phase
            train_loss = self.train_epoch(epoch)
            
            # Validation phase
            valid_loss = self.validate_epoch(epoch)
            
            # Log epoch-level metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            })
            checkpoint_path = os.path.join(self.model_save_path, f"bert2-epoch-{epoch+1}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)  # Save model after each epoch
            # Save the best model
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                model_save_path = os.path.join(self.model_save_path, "bert2-best.pth")
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Saved best model with validation loss: {valid_loss:.4f}")
                
        print("Training completed!")
        wandb.finish()  # Finish wandb run

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        
        # Create iterators for data loaders
        mlm_iter = iter(self.mlm_train_loader)
        anp_iter = iter(self.anp_train_loader)
        
        # Determine the maximum number of batches
        max_batches = max(len(self.mlm_train_loader), len(self.anp_train_loader))
        
        data_iter = tqdm.tqdm(
            range(max_batches),
            desc=f"Epoch {epoch+1} Train",
            total=max_batches,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        
        for i in data_iter:
            # Prepare MLM task batch
            try:
                mlm_batch = next(mlm_iter)
            except StopIteration:
                mlm_iter = iter(self.mlm_train_loader)
                mlm_batch = next(mlm_iter)
                
            # Prepare ANP task batch
            try:
                anp_batch = next(anp_iter)
            except StopIteration:
                anp_iter = iter(self.anp_train_loader)
                anp_batch = next(anp_iter)
            
            # Compute MLM task loss and backpropagate
            mlm_task_dict = self.prepare_task_dict(mlm_batch)
            mlm_loss, _ = self.model(mlm_task_dict)
            
            # Compute ANP task loss and backpropagate
            anp_task_dict = self.prepare_task_dict(anp_batch)
            anp_loss, _ = self.model(anp_task_dict)
            
            total_batch_loss = mlm_loss + anp_loss  # Combine losses FIRST
            # Zero gradients
            self.optim.zero_grad()
            # Use mixed precision for backpropagation
            if self.use_amp:
                self.scaler.scale(total_batch_loss).backward()  # Single backward pass
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                total_batch_loss.backward()  # Single backward
                self.optim.step()
            
            # Record loss
            total_loss += total_batch_loss.item()
            total_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / total_batches
            data_iter.set_postfix(loss=total_batch_loss.item(), avg_loss=avg_loss)

            # Log batch-level metrics to wandb
            if i % self.log_freq == 0:
                wandb.log({
                    "batch": epoch * max_batches + i,
                    "batch_train_loss": total_batch_loss.item(),
                    "batch_avg_train_loss": avg_loss,
                    "lr": self.optim_schedule.get_last_lr()[0]
                })
            
            # Update learning rate
            self.optim_schedule.step()
        
        avg_epoch_loss = total_loss / total_batches
        print(f"Epoch {epoch+1} Train loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        
        # Validate MLM task
        with torch.no_grad():
            for mlm_batch in tqdm.tqdm(
                self.mlm_valid_loader,
                desc=f"Epoch {epoch+1} Valid (MLM)",
                leave=False
            ):
                if "input_ids" in mlm_batch and len(mlm_batch["input_ids"]) == 0:
                    continue
                    
                mlm_task_dict = self.prepare_task_dict(mlm_batch)
                mlm_loss, _ = self.model(mlm_task_dict)
                total_loss += mlm_loss.item()
                total_batches += 1
        
        # Validate ANP task
        with torch.no_grad():
            for anp_batch in tqdm.tqdm(
                self.anp_valid_loader,
                desc=f"Epoch {epoch+1} Valid (ANP)",
                leave=False
            ):
                if "input_a" in anp_batch and len(anp_batch["input_a"]) == 0:
                    continue
                    
                anp_task_dict = self.prepare_task_dict(anp_batch)
                anp_loss, _ = self.model(anp_task_dict)
                total_loss += anp_loss.item()
                total_batches += 1
        
        avg_epoch_loss = total_loss / total_batches if total_batches > 0 else float('inf')
        print(f"Epoch {epoch+1} Valid loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def prepare_task_dict(self, data):
        task_type = data.get("task_type", "mlm")
        task_dict = {"task_type": task_type}
        
        if task_type == "mlm":
            task_dict.update({
                'input_ids': data["input_ids"].to(self.device),
                'labels': data["labels"].to(self.device)
            })
        elif task_type == "anp":
            task_dict.update({
                'input_a': data["input_a"].to(self.device),
                'input_b': data["input_b"].to(self.device),
                'labels': data["labels"].to(self.device)
            })
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
        return task_dict

# Dummy context manager for non-mixed precision training
class dummy_context:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        pass

if __name__ == "__main__":
    data_dir = "."

    tokenizer = AsmTokenizer(
        vocab_file=os.path.join(data_dir, "outputs", f"baseline-vocab.txt")
    )
    print(f"Vocab size: {len(tokenizer.vocab)}")
    # Set cache directory
    cache_dir = os.path.join(data_dir, "outputs", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Load training set (enable memory mapping)
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(data_dir, "outputs", f"baseline-train.jsonl"),
        split="train",
        cache_dir=cache_dir,
        keep_in_memory=False  # Use memory mapping to save memory
    )
    print(f"Train Dataset size: {len(train_dataset)}")

    # Load validation set (enable memory mapping)
    valid_dataset = load_dataset(
        "json",
        data_files=os.path.join(data_dir, "outputs", f"baseline-val.jsonl"),
        split="train",
        cache_dir=cache_dir,
        keep_in_memory=False  # Use memory mapping to save memory
    )
    print(f"Validation Dataset size: {len(valid_dataset)}")
    # Create task-specific datasets
    mlm_train_dataset = TaskDataset(train_dataset, "mlm")
    anp_train_dataset = TaskDataset(train_dataset, "anp")
    mlm_valid_dataset = TaskDataset(valid_dataset, "mlm")
    anp_valid_dataset = TaskDataset(valid_dataset, "anp")
    
    # Create collate functions
    mlm_collate = MLMCollateFn(tokenizer, config.seq_len, train=True)
    anp_collate = ANPCollateFn(tokenizer, config.seq_len)
    combined_collate = CombinedCollateFn(mlm_collate, anp_collate)

    # Create data loaders
    def create_dataloader(dataset, batch_size, collate_fn):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=6,
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
            shuffle=True
        )
    
    mlm_train_loader = create_dataloader(mlm_train_dataset, config.batch_size, combined_collate)
    anp_train_loader = create_dataloader(anp_train_dataset, config.batch_size, combined_collate)
    mlm_valid_loader = create_dataloader(mlm_valid_dataset, config.batch_size, combined_collate)
    anp_valid_loader = create_dataloader(anp_valid_dataset, config.batch_size, combined_collate)
    
    # Create model
    bert_model = BERT2(
        vocab_size=len(tokenizer.vocab),
        d_model=128,
        n_layers=12,
        heads=8,
        seq_len=config.seq_len,
        device=config.device
    )
    # Load pretrained weights
    bert_model.load_state_dict(torch.load(os.path.join(data_dir, "outputs", "bert-pretrain-epoch-8", "bert2-best.pth")))
    # convert model to float32
    bert_model = bert_model.float()

    # Create trainers
    trainer = BERT2PretrainTrainer(
        model=bert_model,
        mlm_train_loader=mlm_train_loader,
        anp_train_loader=anp_train_loader,
        mlm_valid_loader=mlm_valid_loader,
        anp_valid_loader=anp_valid_loader,
        num_epochs=config.epochs,
        model_save_path=config.checkpoint_save_path,
        device=config.device,
        use_amp=config.use_amp
    )
    if config.wandb_run:
        wandb.run.name = config.wandb_run
    trainer.train()
