import argparse
import pickle
import os
from pathlib import Path
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
import gc
from models.bert import BERT2
from models.tokenizer import AsmTokenizer
import numpy as np
from datasets import load_dataset
import json
from models.collatefn import MLM_ANP_CollateFn
from torch.utils.data import TensorDataset


class BERTTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        test_dataloader=None,
        valid_dataloader=None,
        lr=1e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        log_freq=10,
        num_epochs=20,
        model_save_path="",
        device="cuda",
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
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        _ = self.iteration(epoch, self.train_data)
        avg_loss = self.iteration(epoch, self.valid_data, train=False)
        if avg_loss < self.avg_loss:
            self.avg_loss = avg_loss
            torch.save(self.model.state_dict(), self.model_save_path)

    def test(self, epoch):
        _ = self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        mode = "train" if train else "test"
        avg_loss = 0.0

        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            # total=data_loader.dataset._info.dataset_size // data_loader.batch_size,
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        for i, data in data_iter:
            mini_batch_size = 256
            mlm_input_ids = data["mlm_input_ids"]
            mlm_labels = data["mlm_labels"]
            anp_input_ids_a = data["anp_input_ids_a"]
            anp_input_ids_b = data["anp_input_ids_b"]
            anp_input_labels = data["anp_labels"]

            iter_mlm_input_ids = torch.split(mlm_input_ids, mini_batch_size)
            iter_mlm_labels = torch.split(mlm_labels, mini_batch_size)
            iter_anp_input_ids_a = torch.split(anp_input_ids_a, mini_batch_size)
            iter_anp_input_ids_b = torch.split(anp_input_ids_b, mini_batch_size)
            iter_anp_input_labels = torch.split(anp_input_labels, mini_batch_size)
            if train:
                self.model.train()
                # Task 1 mask language model
                loss = 0.0
                for mlm_input_ids, mlm_labels in zip(
                    iter_mlm_input_ids, iter_mlm_labels
                ):
                    mlm_input_ids = mlm_input_ids.to(self.device)
                    mlm_labels = mlm_labels.to(self.device)
                    mlm_loss, _ = self.model.forward_mlm(mlm_input_ids, mlm_labels)
                    self.optim.zero_grad()
                    mlm_loss.backward()
                    self.optim.step()
                    loss += mlm_loss.detach()
                # Task 2. ANP
                for anp_input_ids_a, anp_input_ids_b, anp_input_labels in zip(
                    iter_anp_input_ids_a, iter_anp_input_ids_b, iter_anp_input_labels
                ):
                    anp_input_ids_a = anp_input_ids_a.to(self.device)
                    anp_input_ids_b = anp_input_ids_b.to(self.device)
                    anp_input_labels = anp_input_labels.to(self.device)
                    anp_loss, _ = self.model.forward_anp(
                        anp_input_ids_a, anp_input_ids_b, anp_input_labels
                    )
                    self.optim.zero_grad()
                    anp_loss.backward()
                    self.optim.step()
                    loss += anp_loss.detach()
                self.optim_schedule.step()
            else:
                self.model.eval()
                with torch.no_grad():
                    loss = 0.0
                    for mlm_input_ids, mlm_labels in zip(
                        iter_mlm_input_ids, iter_mlm_labels
                    ):  # Iterate over the DataLoader
                        mlm_input_ids = mlm_input_ids.to(self.device)
                        mlm_labels = mlm_labels.to(self.device)
                        mlm_loss, _ = self.model.forward_mlm(mlm_input_ids, mlm_labels)
                        loss += mlm_loss
                    for anp_input_ids_a, anp_input_ids_b, anp_input_labels in zip(
                        iter_anp_input_ids_a, iter_anp_input_ids_b, iter_anp_input_labels
                    ):
                        anp_input_ids_a = anp_input_ids_a.to(self.device)
                        anp_input_ids_b = anp_input_ids_b.to(self.device)
                        anp_input_labels = anp_input_labels.to(self.device)
                        anp_loss, _ = self.model.forward_anp(
                            anp_input_ids_a, anp_input_ids_b, anp_input_labels
                        )
                        loss += anp_loss

            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item(),
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        print(
            f"EP{epoch}, {mode}: \
            avg_loss={avg_loss / len(data_iter)}"
        )
        return avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--device", default="cuda", dest="device")
    args = parser.parse_args()
    seq_len = 128
    data_dir = "."

    tokenizer = AsmTokenizer(
        vocab_file=os.path.join(data_dir, "outputs", f"baseline-vocab.txt")
    )
    print(f"Vocab size: {len(tokenizer.vocab)}")
    # 设置缓存目录
    cache_dir = os.path.join(data_dir, "outputs", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 加载训练集（启用内存映射）
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(data_dir, "outputs", f"baseline-train.jsonl"),
        split="train",
        cache_dir=cache_dir,
        keep_in_memory=False  # 使用内存映射节省内存
    )
    print(f"Train Dataset size: {len(train_dataset)}")

    # 加载验证集（启用内存映射）
    valid_dataset = load_dataset(
        "json",
        data_files=os.path.join(data_dir, "outputs", f"baseline-val.jsonl"),
        split="train",
        cache_dir=cache_dir,
        keep_in_memory=False  # 使用内存映射节省内存
    )
    print(f"Validation Dataset size: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,  # 添加内存固定
        collate_fn=MLM_ANP_CollateFn(
            tokenizer, seq_len, train=True
        ),  # 使用自定义的collate函数
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=512,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,  # 添加内存固定
        collate_fn=MLM_ANP_CollateFn(
            tokenizer, seq_len, train=True
        ),  # 使用自定义的collate函数
    )

    bert_model = BERT2(vocab_size=len(tokenizer.vocab))

    epochs = 10
    bert_mlm_trainer = BERTTrainer(
        bert_model,
        train_loader,
        valid_dataloader=valid_loader,
        num_epochs=epochs,
        model_save_path=os.path.join(data_dir, "outputs", f"baseline-model"),
        device=args.device,
    )

    for epoch in range(epochs):
        bert_mlm_trainer.train(epoch)
