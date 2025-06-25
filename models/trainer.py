import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.collatefn import MLMCollateFn, ANPCollateFn, BIGCollateFn, GCCollateFn, CombinedCollateFn
import wandb
import tqdm
from models.dataset import TaskDataset
from torch.utils.data import Subset
import torch.nn.functional as F
from models.model import CFGFusionModel

class BERT2PretrainTrainer:
    def __init__(
        self,
        model,
        train_dataset,  # Full training dataset
        valid_dataset,  # Full validation dataset
        tokenizer,
        config
    ):
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)
        self.full_train_dataset = train_dataset  # Save full training dataset
        self.full_valid_dataset = valid_dataset  # Save full validation dataset
        self.tokenizer = tokenizer
        
        # Sampling seed
        self.sample_seed = 42
        
        # Initialize sampling state
        self.train_sampled_indices = set()
        self.train_unsampled_indices = set(range(len(train_dataset)))
        self.valid_sampled_indices = set()
        self.valid_unsampled_indices = set(range(len(valid_dataset)))
        
        # Single optimizer for the whole model
        self.optim = Adam(
            self.model.parameters(), lr=config.lr , betas=config.betas, weight_decay=config.weight_decay
        )
        
        # Calculate total number of iterations
        train_samples_per_epoch = int(len(train_dataset) * config.train_sample_ratio)
        steps_per_epoch = math.ceil(train_samples_per_epoch / config.batch_size)
        total_steps = steps_per_epoch * config.epochs
        
        # Single learning rate scheduler
        # self.optim_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optim,
        #     T_max=total_steps,
        #     eta_min=1e-6  # minimum learning rate
        # )
        self.optim_schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=1e-3,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            final_div_factor=1e2,
        )

        self.log_freq = config.log_freq
        self.best_loss = float('inf')
        self.model_save_path = config.checkpoint_save_path
        self.epochs = config.epochs
        self.use_amp = config.use_amp
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        if config.use_wandb:
            os.environ["WANDB_MODE"] = "online"  # Enable Weights & Biases logging
        else:
            os.environ["WANDB_MODE"] = "disabled"  # Disable Weights & Biases logging
        # Initialize wandb
        wandb.init(
            project=config.wandb_project, 
            name=config.wandb_run,
            config={
                "learning_rate": config.lr,
                "weight_decay": config.weight_decay,
                "batch_size": config.batch_size,
                "epochs": self.epochs,
                "device": self.device,
                "train_sample_ratio": config.train_sample_ratio,
                "val_sample_ratio": config.val_sample_ratio,
                "total_training_steps": total_steps
            }
        )
        wandb.watch(self.model, log=None)
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Create collate functions
        self.mlm_collate = MLMCollateFn(
            tokenizer, 
            config.seq_len, 
            train=True
            )
        self.anp_collate = ANPCollateFn(
            tokenizer, 
            config.seq_len
            )
        self.combined_collate = CombinedCollateFn(self.mlm_collate, self.anp_collate)
    def create_subset(self, dataset, indices):
        """Create dataset subset (compatible with custom datasets)"""
        return Subset(dataset, indices) 
    def create_dataloader(self, dataset, batch_size, shuffle=True):
        """Create DataLoader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=6,
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=self.combined_collate,
            shuffle=shuffle
        )
    
    def sample_dataset_indices(self, dataset, sample_ratio, dataset_type):
        """Non-repetitive sampling, continue sampling after reset"""
        if dataset_type == "training":
            unsampled = self.train_unsampled_indices
            sampled = self.train_sampled_indices
        else:
            unsampled = self.valid_unsampled_indices
            sampled = self.valid_sampled_indices

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

    def train(self):
        for epoch in range(self.epochs):
            # Set random seed for reproducibility
            self.sample_seed = epoch + 1  # Use a different seed for each epoch
            torch.manual_seed(self.sample_seed)
            np.random.seed(self.sample_seed)
            random.seed(self.sample_seed)
            
            # Sample training set
            train_indices = self.sample_dataset_indices(
                self.full_train_dataset, 
                self.config.train_sample_ratio,
                "training"
            )
            train_subset = self.create_subset(self.full_train_dataset, train_indices)
            
            # Create task-specific datasets
            mlm_train_dataset = TaskDataset(train_subset, "mlm")
            anp_train_dataset = TaskDataset(train_subset, "anp")
            
            # Create data loaders
            mlm_train_loader = self.create_dataloader(mlm_train_dataset, self.config.batch_size)
            anp_train_loader = self.create_dataloader(anp_train_dataset, self.config.batch_size)
            
            # Training phase
            train_loss = self.train_epoch(epoch, mlm_train_loader, anp_train_loader)
            
            # Sample validation set
            valid_indices = self.sample_dataset_indices(
                self.full_valid_dataset, 
                self.config.val_sample_ratio,
                "validation"
            )
            valid_subset = self.create_subset(self.full_valid_dataset, valid_indices)
            
            # Create task-specific datasets
            mlm_valid_dataset = TaskDataset(valid_subset, "mlm")
            anp_valid_dataset = TaskDataset(valid_subset, "anp")
            
            # Create data loaders
            mlm_valid_loader = self.create_dataloader(mlm_valid_dataset, self.config.batch_size, shuffle=False)
            anp_valid_loader = self.create_dataloader(anp_valid_dataset, self.config.batch_size, shuffle=False)
            
            # Validation phase
            valid_loss = self.validate_epoch(epoch, mlm_valid_loader, anp_valid_loader)
            
            # Log epoch metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            })
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.model_save_path, f"bert2-epoch-{epoch+1}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            
            # Save the best model
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                model_save_path = os.path.join(self.model_save_path, "bert2-best.pth")
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Saved best model with validation loss: {valid_loss:.4f}")
                
        print("Training completed!")
        wandb.finish()  # Finish wandb run

    def train_epoch(self, epoch, mlm_train_loader, anp_train_loader):
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        
        # Create iterators for data loaders
        mlm_iter = iter(mlm_train_loader)
        anp_iter = iter(anp_train_loader)
        
        # Determine the maximum number of batches
        max_batches = max(len(mlm_train_loader), len(anp_train_loader))
        
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
                mlm_iter = iter(mlm_train_loader)
                mlm_batch = next(mlm_iter)
                
            # Prepare ANP task batch
            try:
                anp_batch = next(anp_iter)
            except StopIteration:
                anp_iter = iter(anp_train_loader)
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
                    "mlm_loss": mlm_loss.item(),
                    "anp_loss": anp_loss.item(),
                    "batch_train_loss": total_batch_loss.item(),
                    "batch_avg_train_loss": avg_loss,
                    "lr": self.optim_schedule.get_last_lr()[0]
                })
            
            # Update learning rate
            self.optim_schedule.step()
        
        avg_epoch_loss = total_loss / total_batches
        print(f"Epoch {epoch+1} Train loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def validate_epoch(self, epoch, mlm_valid_loader, anp_valid_loader):
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        
        # Validate MLM task
        with torch.no_grad():
            for mlm_batch in tqdm.tqdm(
                mlm_valid_loader,
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
                anp_valid_loader,
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
        task_dict.update({
            'input_ids': data["input_ids"].to(self.device),
            'segment_ids': data.get("segment_ids", None).to(self.device) if "segment_ids" in data else None,
            'labels': data["labels"].to(self.device)
        })        
        return task_dict

class BERT4PretrainTrainer(BERT2PretrainTrainer):
    def __init__(
        self,
        model,
        train_dataset,  # Full training dataset
        valid_dataset,  # Full validation dataset
        tokenizer,
        config
    ):
        super().__init__(model, train_dataset, valid_dataset, tokenizer, config)
        # Add collate functions for new tasks
        self.big_collate = BIGCollateFn(
            tokenizer, 
            config.seq_len,
        )
        self.gc_collate = GCCollateFn(
            tokenizer, 
            config.seq_len,
        )
        
        # Update combined_collate to support four tasks
        self.combined_collate = CombinedCollateFn(
            self.mlm_collate, 
            self.anp_collate,
            self.big_collate,
            self.gc_collate
        )
        # 重新计算总步数（每个batch两次更新）
        train_samples_per_epoch = int(len(train_dataset) * config.train_sample_ratio)
        steps_per_epoch = math.ceil(train_samples_per_epoch / config.batch_size)
        total_steps = 2 * steps_per_epoch * config.epochs  # 乘以2因为每个batch两次更新
        
        # 重新初始化学习率调度器
        self.optim_schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=1e-3,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            final_div_factor=1e2,
        )
        self.loss_weights = {
            "mlm": 1.0,
            "anp": 1.0,
            "big": 1.0,
            "gc": 1.0
        }
        wandb.config.update({
            "mlm_loss_weight": self.loss_weights["mlm"],
            "anp_loss_weight": self.loss_weights["anp"],
            "big_loss_weight": self.loss_weights["big"],
            "gc_loss_weight": self.loss_weights["gc"]
        })
    def create_dataloader(self, dataset, batch_size, shuffle=True):
        """Create DataLoader (override parent method)"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=6,
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=self.combined_collate,
            shuffle=shuffle
        )
    
    def train(self):
        for epoch in range(self.epochs):
            # Set random seed
            self.sample_seed = epoch + 1
            torch.manual_seed(self.sample_seed)
            np.random.seed(self.sample_seed)
            random.seed(self.sample_seed)
            
            # Sample training set
            train_indices = self.sample_dataset_indices(
                self.full_train_dataset, 
                self.config.train_sample_ratio,
                "training"
            )
            # Create subset in a compatible way
            train_subset = self.create_subset(self.full_train_dataset, train_indices)
            
            # Create task datasets (add BIG and GC tasks)
            mlm_train_dataset = TaskDataset(train_subset, "mlm")
            anp_train_dataset = TaskDataset(train_subset, "anp")
            big_train_dataset = TaskDataset(train_subset, "big")
            gc_train_dataset = TaskDataset(train_subset, "gc")
            
            # Create data loaders
            mlm_train_loader = self.create_dataloader(mlm_train_dataset, self.config.batch_size)
            anp_train_loader = self.create_dataloader(anp_train_dataset, self.config.batch_size)
            big_train_loader = self.create_dataloader(big_train_dataset, self.config.batch_size)
            gc_train_loader = self.create_dataloader(gc_train_dataset, self.config.batch_size)
            
            # Training phase
            train_loss = self.train_epoch(
                epoch, 
                mlm_train_loader, 
                anp_train_loader,
                big_train_loader,
                gc_train_loader
            )
            
            # Sample validation set
            valid_indices = self.sample_dataset_indices(
                self.full_valid_dataset, 
                self.config.val_sample_ratio,
                "validation"
            )
            # Create subset in a compatible way
            valid_subset = self.create_subset(self.full_valid_dataset, valid_indices)
            
            # Create validation task datasets
            mlm_valid_dataset = TaskDataset(valid_subset, "mlm")
            anp_valid_dataset = TaskDataset(valid_subset, "anp")
            big_valid_dataset = TaskDataset(valid_subset, "big")
            gc_valid_dataset = TaskDataset(valid_subset, "gc")
            
            # Create validation data loaders
            mlm_valid_loader = self.create_dataloader(mlm_valid_dataset, self.config.batch_size, shuffle=False)
            anp_valid_loader = self.create_dataloader(anp_valid_dataset, self.config.batch_size, shuffle=False)
            big_valid_loader = self.create_dataloader(big_valid_dataset, self.config.batch_size, shuffle=False)
            gc_valid_loader = self.create_dataloader(gc_valid_dataset, self.config.batch_size, shuffle=False)
            
            # Validation phase
            valid_loss = self.validate_epoch(
                epoch, 
                mlm_valid_loader, 
                anp_valid_loader,
                big_valid_loader,
                gc_valid_loader
            )
            
            # Log to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            })
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.model_save_path, f"bert4-epoch-{epoch+1}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            
            # Save best model
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                model_save_path = os.path.join(self.model_save_path, "bert4-best.pth")
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Saved best model with validation loss: {valid_loss:.4f}")
                
        print("BERT4 training completed!")
        wandb.finish()

    def train_epoch(self, epoch, mlm_train_loader, anp_train_loader, big_train_loader, gc_train_loader):
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        # Create iterators
        mlm_iter = iter(mlm_train_loader)
        anp_iter = iter(anp_train_loader)
        big_iter = iter(big_train_loader)
        gc_iter = iter(gc_train_loader)

        # Determine the maximum number of batches
        max_batches = max(
            len(mlm_train_loader),
            len(anp_train_loader),
            len(big_train_loader),
            len(gc_train_loader)
        )

        data_iter = tqdm.tqdm(
            range(max_batches),
            desc=f"Epoch {epoch+1} Train",
            total=max_batches,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )

        # Check whether to freeze MLM gradients
        freeze_mlm = epoch > self.epochs // 2

        for i in data_iter:
            # ====================== Phase 1: MLM, ANP, BIG tasks ======================
            task_losses = []
            mlm_loss = anp_loss = big_loss = gc_loss = 0

            # Handle MLM task
            try:
                mlm_batch = next(mlm_iter)
                mlm_task_dict = self.prepare_task_dict(mlm_batch)

                if freeze_mlm:
                    # Freeze MLM gradients
                    with torch.no_grad():
                        mlm_loss, _ = self.model(mlm_task_dict)
                else:
                    # Train MLM normally
                    mlm_loss, _ = self.model(mlm_task_dict)

                task_losses.append(mlm_loss * self.loss_weights["mlm"])
            except StopIteration:
                pass

            # Handle ANP task
            try:
                anp_batch = next(anp_iter)
                anp_task_dict = self.prepare_task_dict(anp_batch)
                anp_loss, _ = self.model(anp_task_dict)
                task_losses.append(anp_loss * self.loss_weights["anp"])
            except StopIteration:
                pass

            # Handle BIG task
            try:
                big_batch = next(big_iter)
                big_task_dict = self.prepare_task_dict(big_batch)
                big_loss, _ = self.model(big_task_dict)
                task_losses.append(big_loss * self.loss_weights["big"])
            except StopIteration:
                pass

            # Compute phase 1 loss and update
            if task_losses:
                phase1_loss = sum(task_losses)

                # Zero gradients
                self.optim.zero_grad()

                # Mixed precision training
                if self.use_amp:
                    self.scaler.scale(phase1_loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    phase1_loss.backward()
                    self.optim.step()

                # Update learning rate
                self.optim_schedule.step()

            # ====================== Phase 2: GC task ======================
            try:
                gc_batch = next(gc_iter)
                gc_task_dict = self.prepare_task_dict(gc_batch)
                gc_loss, _ = self.model(gc_task_dict)
                gc_loss_weighted = gc_loss * self.loss_weights["gc"]

                # Zero gradients
                self.optim.zero_grad()

                # Mixed precision training
                if self.use_amp:
                    self.scaler.scale(gc_loss_weighted).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    gc_loss_weighted.backward()
                    self.optim.step()

                # Update learning rate
                self.optim_schedule.step()
            except StopIteration:
                gc_loss_weighted = 0

            # ====================== Compute total loss ======================
            # Compute total batch loss (for logging)
            batch_loss = phase1_loss.item() if task_losses else 0
            batch_loss += gc_loss_weighted.item() if gc_loss_weighted != 0 else 0

            # Accumulate loss
            total_loss += batch_loss
            total_batches += 1

            # Update progress bar
            avg_loss = total_loss / total_batches
            data_iter.set_postfix(loss=batch_loss, avg_loss=avg_loss)

            # Log to wandb
            if i % self.log_freq == 0:
                log_data = {
                    "batch": epoch * max_batches + i,
                    "batch_train_loss": batch_loss,
                    "batch_avg_train_loss": avg_loss,
                    "lr": self.optim_schedule.get_last_lr()[0],
                    "freeze_mlm": int(freeze_mlm)
                }

                # Log each task's loss
                if mlm_loss != 0:
                    log_data["mlm_loss"] = mlm_loss.item()
                if anp_loss != 0:
                    log_data["anp_loss"] = anp_loss.item()
                if big_loss != 0:
                    log_data["big_loss"] = big_loss.item()
                if gc_loss != 0:
                    log_data["gc_loss"] = gc_loss.item()

                wandb.log(log_data)

        avg_epoch_loss = total_loss / total_batches if total_batches > 0 else 0.0
        print(f"Epoch {epoch+1} Train loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def validate_epoch(self, epoch, mlm_valid_loader, anp_valid_loader, big_valid_loader, gc_valid_loader):
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
    
        # Validate MLM task
        with torch.no_grad():
            for mlm_batch in mlm_valid_loader:
                if "input_ids" in mlm_batch and len(mlm_batch["input_ids"]) == 0:
                    continue
                    
                mlm_task_dict = self.prepare_task_dict(mlm_batch)
                mlm_loss, _ = self.model(mlm_task_dict)
                total_loss += mlm_loss.item()
                total_batches += 1
        
        # Validate ANP task
        with torch.no_grad():
            for anp_batch in anp_valid_loader:
                if "input_ids" in anp_batch and len(anp_batch["input_ids"]) == 0:
                    continue
                    
                anp_task_dict = self.prepare_task_dict(anp_batch)
                anp_loss, _ = self.model(anp_task_dict)
                total_loss += anp_loss.item()
                total_batches += 1
        
        # Validate BIG task
        with torch.no_grad():
            for big_batch in big_valid_loader:
                if "input_ids" in big_batch and len(big_batch["input_ids"]) == 0:
                    continue
                    
                big_task_dict = self.prepare_task_dict(big_batch)
                big_loss, _ = self.model(big_task_dict)
                total_loss += big_loss.item()
                total_batches += 1
        
        # Validate GC task
        with torch.no_grad():
            for gc_batch in gc_valid_loader:
                if "input_ids" in gc_batch and len(gc_batch["input_ids"]) == 0:
                    continue
                    
                gc_task_dict = self.prepare_task_dict(gc_batch)
                gc_loss, _ = self.model(gc_task_dict)
                total_loss += gc_loss.item()
                total_batches += 1
        
        avg_epoch_loss = total_loss / total_batches if total_batches > 0 else float('inf')
        print(f"Epoch {epoch+1} Valid loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss
    

class BERTFinetuneTrainer:
    def __init__(
        self,
        model: CFGFusionModel,
        train_dataset,  # Full training dataset
        val_dataset,    # Full validation dataset
        config
    ):
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)
        self.train_dataset = train_dataset  # Save full training dataset
        self.val_dataset = val_dataset      # Save full validation dataset
        self.log_freq = config.log_freq
        self.num_epochs = config.epochs
        self.model_save_path = config.checkpoint_save_path
        self.use_amp = config.use_amp
        self.best_accuracy = 0
        self.best_val_loss = float('inf')
        
        # DataLoader parameters
        self.batch_size = config.batch_size
        
        # Sampling seed
        self.sample_seed = 42
        
        # Initialize sampling state
        self.train_sampled_indices = set()
        self.train_unsampled_indices = set(range(len(train_dataset)))
        self.val_sampled_indices = set()
        self.val_unsampled_indices = set(range(len(val_dataset)))
        
        # Optimizer and loss function
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=config.lr, 
            betas=config.betas, 
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CosineEmbeddingLoss(margin=0.0)
        
        # Calculate total global steps
        train_samples_per_epoch = int(len(train_dataset) * config.train_sample_ratio)
        steps_per_epoch = math.ceil(train_samples_per_epoch / config.batch_size)
        total_steps = steps_per_epoch * config.epochs

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
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
            
        # Initialize wandb if enabled
        if config.use_wandb:
            os.environ["WANDB_MODE"] = "online"
            wandb.init(
                project=config.wandb_project,
                config={
                    "batch_size": config.batch_size,
                    "learning_rate": config.lr,
                    "epochs": config.epochs,
                    "device": config.device,
                    "train_sample_ratio": config.train_sample_ratio,
                    "val_sample_ratio": config.val_sample_ratio,
                    "total_training_steps": total_steps
                }
            )
            wandb.run.name = config.wandb_run
            wandb.watch(self.model, log=None)
        else:
            os.environ["WANDB_MODE"] = "disabled"
        
        # Ensure output directory exists
        os.makedirs(config.checkpoint_save_path, exist_ok=True)

    def train(self):
        print(f"Starting training on {self.device}...")
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Set random seed for reproducibility
            self.sample_seed = epoch + 1
            torch.manual_seed(self.sample_seed)
            np.random.seed(self.sample_seed)
            random.seed(self.sample_seed)
            
            # Sample training set
            train_indices = self.sample_dataset_indices(
                self.train_dataset, 
                self.config.train_sample_ratio,
                "training"
            )
            
            # Sample validation set
            val_indices = self.sample_dataset_indices(
                self.val_dataset, 
                self.config.val_sample_ratio,
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
            if self.config.use_wandb:
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
        if self.config.use_wandb:
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
            num_workers=6,
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=True
        )

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        total_loss = 0
        progress = tqdm.tqdm(
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
                a_embeddings = self.model(a_ids, a_adj)
                t_embeddings = self.model(t_ids, t_adj)
                loss = self.criterion(a_embeddings, t_embeddings, labels.float())
            
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
            
            if i % self.log_freq == 0 and self.config.use_wandb:
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
        progress = tqdm.tqdm(
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
                a_embeddings = self.model(a_ids, a_adj)
                t_embeddings = self.model(t_ids, t_adj)
                loss = self.criterion(a_embeddings, t_embeddings, labels.float())
                
                # Calculate accuracy
                cosine_sim = F.cosine_similarity(a_embeddings, t_embeddings, dim=1)
                idx = cosine_sim > 0
                predictions = torch.zeros_like(cosine_sim)
                predictions[idx] = 1
                predictions[~idx] = -1
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