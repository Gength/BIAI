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
from models.dataset import BERTDataset
from models.bert import BERT
from models.tokenizer import AsmTokenizer
import numpy as np
from datasets import load_dataset
import json
from models.collatefn import CollateFn

class BERTTrainer:
	def __init__(
		self, 
		model, 
		train_dataloader, 
		test_dataloader=None, 
		valid_dataloader=None,
		lr= 1e-5,
		weight_decay=0.01,
		betas=(0.9, 0.999),
		log_freq=10,
		num_epochs=20,
		model_save_path="",
		device='cuda'
		):

		self.device = device
		self.model = model.to(device)
		self.train_data = train_dataloader
		self.test_data = test_dataloader
		self.valid_data = valid_dataloader

		self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
		self.optim_schedule = torch.optim.lr_scheduler.OneCycleLR(
    		self.optim,
    		max_lr=1e-3,
    		steps_per_epoch=train_dataloader.dataset._info.dataset_size // train_dataloader.batch_size,
    		epochs=num_epochs,
    		pct_start=0.1,
    		anneal_strategy='cos',
    		final_div_factor=1e2
		)

		# Using Negative Log Likelihood Loss function for predicting the masked_token
		self.criterion = torch.nn.NLLLoss(ignore_index=0)
		self.log_freq = log_freq
		self.avg_loss = 999999
		self.model_save_path = model_save_path
		print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
	
	def train(self, epoch):
		_ = self.iteration(epoch, self.train_data)
		avg_loss = self.iteration(epoch, self.valid_data, train=False)
		if(avg_loss < self.avg_loss):
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
			total=data_loader.dataset._info.dataset_size // data_loader.batch_size,
			bar_format="{l_bar}{r_bar}"
		)

		for i, data in data_iter:
      		# 0. batch_data will be sent into the device(GPU or cpu)
			data = {key: value.to(self.device) for key, value in data.items()}
			if(train):
				self.model.train()
				# 1. forward the masked_lm model
				mask_lm_output = self.model.forward(data["bert_input"])			

				# 2-1. NLLLoss of predicting masked token word
				# loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])	
				loss = self.criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))			
				# 3. backward and optimization only in train
				#if train:
				self.optim.zero_grad()
				loss.backward()
				self.optim.step()
				self.optim_schedule.step()
			else:
				self.model.eval()
				with torch.no_grad():
					mask_lm_output = self.model.forward(data["bert_input"])
					loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

			avg_loss += loss.item()

			post_fix = {
				"epoch": epoch,
				"iter": i,
				"avg_loss": avg_loss / (i + 1),
				"loss": loss.item()
			}

			if i % self.log_freq == 0:
				data_iter.write(str(post_fix))
		print(
			f"EP{epoch}, {mode}: \
			avg_loss={avg_loss / len(data_iter)}"
		) 
		return avg_loss

def load_assembly_data(data):
	data_pairs = []
	for item in data:
		for i in range(0, len(item)-1, 2):
			data_pairs.append((item[i].strip(), item[i+1].strip()))
	return data_pairs

if __name__=="__main__":
	parser = argparse.ArgumentParser(description="Command line parameters")
	parser.add_argument('--device', default="cuda", dest="device")
	args = parser.parse_args()
	seq_len = 16
	data_dir = "."

	tokenizer = AsmTokenizer(vocab_file=os.path.join(data_dir, "outputs", f"baseline-vocab.txt"))
	print(f"Vocab size: {len(tokenizer.vocab)}")

	train_dataset_path = os.path.join(data_dir, "outputs", f"baseline-train.jsonl")
	train_metadata_path = os.path.join(data_dir, "outputs", f"baseline-train-metadata.jsonl")
	train_dataset = load_dataset('json', data_files=train_dataset_path, split='train', streaming=True)
	with open(train_metadata_path, "r", encoding="utf-8") as f:
		metadata = json.load(f)
		dataset_size = metadata['__metadata__']['dataset_size']
		print(f"Train Dataset size: {dataset_size}")
		train_dataset.info.dataset_size = dataset_size
	shuffled_train_dataset = train_dataset.shuffle(seed=42, buffer_size=5000)

	valid_dataset_path = os.path.join(data_dir, "outputs", f"baseline-valid.jsonl")
	valid_metadata_path = os.path.join(data_dir, "outputs", f"baseline-valid-metadata.jsonl")
	valid_dataset = load_dataset('json', data_files=valid_dataset_path, split='train', streaming=True)
	with open(valid_metadata_path, "r", encoding="utf-8") as f:
		metadata = json.load(f)
		dataset_size = metadata['__metadata__']['dataset_size']
		print(f"Validation Dataset size: {dataset_size}")
		valid_dataset.info.dataset_size = dataset_size
	shuffled_valid_dataset = valid_dataset.shuffle(seed=42, buffer_size=5000)

	train_loader = DataLoader(
		shuffled_train_dataset, 
		batch_size=512, 
		num_workers=12,
		prefetch_factor=4,
		persistent_workers=True,
		pin_memory=True,  # 添加内存固定
		collate_fn=CollateFn(tokenizer, seq_len=seq_len, train=True)

		)
	valid_loader = DataLoader(
		shuffled_valid_dataset, 
		batch_size=512, 
		num_workers=12,
		prefetch_factor=4,
		persistent_workers=True,
		pin_memory=True,  # 添加内存固定
		collate_fn=CollateFn(tokenizer, seq_len=seq_len, train=True)
		)
	
	bert_model = BERT(
	  vocab_size=len(tokenizer.vocab),
	  d_model=128,
	  n_layers=2,
	  heads=2,
	  dropout=0.2,
	  device=args.device
	)

	epochs = 10
	bert_trainer = BERTTrainer(bert_model, train_loader, valid_dataloader=valid_loader, num_epochs=epochs, model_save_path=os.path.join(data_dir, "outputs", f"baseline-model"), device=args.device)


	for epoch in range(epochs):
		bert_trainer.train(epoch)
