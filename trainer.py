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
from torch.amp import autocast
class BERTTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        test_dataloader=None,
        valid_dataloader=None,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        log_freq=10,
        num_epochs=20,
        model_save_path="",
        device="cuda",
        use_amp=True,
    ):

        self.device = device
        self.model = model.to(device)
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.valid_data = valid_dataloader

        self.optim = Adam(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        self.optim_schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=1e-3,
            # steps_per_epoch=train_dataloader.dataset._info.dataset_size
            # // train_dataloader.batch_size,
            steps_per_epoch=len(train_dataloader),
            epochs=num_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=1e2,
        )

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        # self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
        self.avg_loss = 999999
        self.model_save_path = model_save_path
        self.use_amp = use_amp
        if use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        train_loss = self.iteration(epoch, self.train_data, train=True)
        valid_loss = self.iteration(epoch, self.valid_data, train=False)
        
        if valid_loss < self.avg_loss:
            self.avg_loss = valid_loss
            torch.save(self.model.state_dict(), self.model_save_path)
            print(f"Saved best model with validation loss: {valid_loss:.4f}")
        
        return train_loss, valid_loss

    def test(self, epoch):
        _ = self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        mode = "train" if train else "valid"
        total_loss = 0.0
        total_batches = 0

        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc=f"Epoch {epoch+1} {mode}",
            total=len(data_loader),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )

        for i, data in data_iter:
            # Skip empty batches
            if "input_ids" in data and len(data["input_ids"]) == 0:
                continue
            if "input_a" in data and len(data["input_a"]) == 0:
                continue
                
            # Prepare task_dict based on task type
            task_dict = self.prepare_task_dict(data)
            
            if train:
                self.model.train()
                self.optim.zero_grad()
                
                # Mixed precision training
                if self.use_amp:
                    with autocast('cuda', dtype=torch.float16):
                        loss, _ = self.model(task_dict)
                    # Scale the loss and perform backpropagation
                    self.scaler.scale(loss).backward()
                    # Unscale gradients and update parameters
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    loss, _ = self.model(task_dict)
                    loss.backward()
                    self.optim.step()
                
                self.optim_schedule.step()
            else:
                self.model.eval()
                with torch.no_grad():
                    loss, _ = self.model(task_dict)
                
            total_loss += loss.item()
            total_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / total_batches if total_batches > 0 else 0
            data_iter.set_postfix(loss=loss.item(), avg_loss=avg_loss)
        
        avg_epoch_loss = total_loss / total_batches if total_batches > 0 else float('inf')
        print(f"Epoch {epoch+1} {mode} loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss
    
    def prepare_task_dict(self, data):
        """
        Prepare the task dictionary based on the input data.
        This method is used to handle different task types (MLM, ANP).
        """
        task_type = data.get("task_type", "mlm")
        if task_type == "mlm":
            task_dict = {
                'task_type': 'mlm',
                'input_ids': data["input_ids"].to(self.device),
                'labels': data["labels"].to(self.device)
            }
        elif task_type == "anp":
            task_dict = {
                'task_type': 'anp',
                'input_a': data["input_a"].to(self.device),
                'input_b': data["input_b"].to(self.device),
                'labels': data["labels"].to(self.device)
            }
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        return task_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--device", default="cuda", dest="device")
    parser.add_argument("--epochs", type=int, default=20, dest="epochs")
    parser.add_argument("--batch_size", type=int, default=256, dest="batch_size")
    args = parser.parse_args()
    seq_len = 128
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
    mlm_collate = MLMCollateFn(tokenizer, seq_len, train=True)
    anp_collate = ANPCollateFn(tokenizer, seq_len)
    combined_collate = CombinedCollateFn(mlm_collate, anp_collate)

    # Create data loaders
    def create_dataloader(dataset, batch_size, collate_fn):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
            shuffle=True
        )
    
    mlm_train_loader = create_dataloader(mlm_train_dataset, args.batch_size, combined_collate)
    anp_train_loader = create_dataloader(anp_train_dataset, args.batch_size, combined_collate)
    mlm_valid_loader = create_dataloader(mlm_valid_dataset, args.batch_size, combined_collate)
    anp_valid_loader = create_dataloader(anp_valid_dataset, args.batch_size, combined_collate)
    
    # Create model
    bert_model = BERT2(
        vocab_size=len(tokenizer.vocab),
        d_model=128,
        n_layers=12,
        heads=8,
        seq_len=seq_len,
        device=args.device
    )

    # Create trainers
    mlm_trainer = BERTTrainer(
        bert_model,
        mlm_train_loader,
        valid_dataloader=mlm_valid_loader,
        num_epochs=args.epochs,
        model_save_path=os.path.join(data_dir, "outputs", f"mlm-model.pth"),
        device=args.device,
    )
    
    anp_trainer = BERTTrainer(
        bert_model,
        anp_train_loader,
        valid_dataloader=anp_valid_loader,
        num_epochs=args.epochs,
        model_save_path=os.path.join(data_dir, "outputs", f"anp-model.pth"),
        device=args.device,
    )

    # Training loop - alternate training tasks
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} === [MLM Task]")
        mlm_train_loss, mlm_valid_loss = mlm_trainer.train(epoch)
        
        print(f"\n=== Epoch {epoch+1}/{args.epochs} === [ANP Task]")
        anp_train_loss, anp_valid_loss = anp_trainer.train(epoch)
        
        # Save combined model (including both tasks)
        torch.save(bert_model.state_dict(), os.path.join(data_dir, "outputs", f"combined-model-epoch{epoch}.pth"))
        print(f"Saved combined model for epoch {epoch+1}")
    
    print("Training completed!")
