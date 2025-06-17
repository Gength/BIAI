import torch
from torch.utils.data import Dataset
import random
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from models.tokenizer import AsmTokenizer
from utils.utility import tokenize_and_pad
class BERTMLMDataset(Dataset):
	def __init__(self, instr_blocks, tokenizer:AsmTokenizer, max_len=128, train=True):
		self.tokenizer = tokenizer
		self.max_len = max_len
		self.instr_blocks = instr_blocks
		self.train = train


	def __len__(self):
		return len(self.instr_blocks)
		
	def random_mask(self, ids):
		output = []
		labels = []
		for id in ids:
			if id == self.tokenizer.sep_token_id:
				output.append(id)
				labels.append(0)
			if random.random() < 0.15:
				rand_val = random.random()
				if rand_val < 0.8:
					output.append(self.tokenizer.mask_token_id)  # 80% Replace with MASK
				elif rand_val < 0.9:
					output.append(random.choice(list(self.tokenizer.vocab.values())))  # 10% Random token
				else:
					output.append(id)  # 10% Keep original
				labels.append(id)
			else:
				output.append(id)
				labels.append(0)
		assert(len(output) == len(labels))
		return output, labels
		
	def __getitem__(self, idx):
		text = self.instr_blocks[idx]
		ids: list = self.tokenizer.encode(text)
		if self.train:
			ids, labels = self.random_mask(ids)
		else:
			labels = ids.copy()
		# Reserve [CLS] and [SEP], truncate original tokens
		ids = ids[:self.max_len - 2]
		labels = labels[:self.max_len - 2]
		pad_len = self.max_len - len(ids) - 2  # minus <CLS> and <SEP>
		ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
		labels = [0] + labels + [0]

		ids += [self.tokenizer.pad_token_id] * pad_len
		labels += [0] * pad_len

		ids = torch.tensor(ids, dtype=torch.long)
		labels = torch.tensor(labels, dtype=torch.long)
		return ids, labels

class BERTANPDataset(Dataset):
	def __init__(self, instr_blocks, tokenizer:AsmTokenizer, adj, max_len=128):
		self.tokenizer = tokenizer
		self.adj = lil_matrix(tuple(adj['shape']), dtype=np.int32)
		self.adj[adj['row'], adj['col']] = adj['data']
		self.max_len = max_len
		self.instr_blocks = instr_blocks

		positive_pairs = list(zip(adj['row'], adj['col']))
		block_ids = np.array([i for i in range(adj['shape'][0])])
		negative_pairs = []
		while len(negative_pairs) < len(positive_pairs):
			i = random.choice(block_ids)
			j = random.choice(block_ids)
			if (i, j) not in positive_pairs:
				negative_pairs.append((i, j))
		self.pairs = positive_pairs + negative_pairs

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, idx):
		text_a_idx, text_b_idx = self.pairs[idx]
		text_a = self.instr_blocks[text_a_idx]
		text_b = self.instr_blocks[text_b_idx]

		ids_a = tokenize_and_pad(text_a, self.tokenizer, self.max_len)
		ids_b = tokenize_and_pad(text_b, self.tokenizer, self.max_len)

		ids_a = torch.tensor(ids_a, dtype=torch.long)
		ids_b = torch.tensor(ids_b, dtype=torch.long)

		label = torch.tensor(self.adj[text_a_idx, text_b_idx], dtype=torch.long)
		return ids_a, ids_b, label

class TaskDataset(Dataset):
	'''Wrap the dataset and add task type information'''
	def __init__(self, dataset, task_type):
		self.dataset = dataset
		self.task_type = task_type
		
	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		item = self.dataset[idx]
		item["task_type"] = self.task_type
		return item