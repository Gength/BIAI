import torch
from torch.utils.data import Dataset
import random
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from models.tokenizer import AsmTokenizer
from utils.utility import tokenize_and_pad
from datasets import load_dataset
import pickle
import os
from scipy.sparse import coo_matrix
import heapq
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

# Custom dataset class
class FunctionPairDataset(Dataset):
	def __init__(self, csv_path, jsonl_path, mapping_path, tokenizer:AsmTokenizer, seq_len=128, max_nodes=300):
		self.df = pd.read_csv(csv_path)
		self.dataset = load_dataset(
			"json", 
			data_files=jsonl_path,
			split="train",
			cache_dir=os.path.join(".", "outputs", "cache"),
			keep_in_memory=False
		)
		with open(mapping_path, "rb") as f:
			self.mapping = pickle.load(f)
		self.tokenizer = tokenizer
		self.seq_len = seq_len
		self.max_nodes = max_nodes

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
		instr_blocks = func_data["instruction_blocks"]
		adj_data = func_data["adjacency_matrix"]
		n_blocks = adj_data["shape"][0]
		
		# Create sparse adjacency matrix
		adj = coo_matrix(
			(adj_data["data"], (adj_data["row"], adj_data["col"])),
			shape=adj_data["shape"]
		)
		
		# === Skip clustering, process the original graph directly ===
		return self._process_with_padding(instr_blocks, adj, n_blocks)

	def select_key_nodes(self, adj_matrix, actual_nodes, max_nodes):
		"""Select key nodes based on node importance (sparse matrix implementation)"""
		if actual_nodes <= max_nodes:
			# Not enough nodes, return all nodes directly
			return np.arange(actual_nodes)
		
		# Ensure adj_matrix is in COO format
		if not isinstance(adj_matrix, coo_matrix):
			adj_matrix = adj_matrix.tocoo()
		
		# Calculate node importance (in-degree + out-degree)
		row, col = adj_matrix.row, adj_matrix.col
		in_degrees = np.bincount(col, minlength=actual_nodes)
		out_degrees = np.bincount(row, minlength=actual_nodes)
		importance_scores = in_degrees + out_degrees
		
		# Ensure the entry node (index 0) is included
		importance_scores[0] = 10**18  # Use a sufficiently large integer to ensure selection
		
		# Sort by importance and select top-k nodes
		# Use heap sort for efficiency (O(n log k))
		selected_indices = heapq.nlargest(
			max_nodes, 
			range(actual_nodes), 
			key=importance_scores.__getitem__
		)
		selected_indices.sort()  # Keep original order
		
		return selected_indices

	def _process_with_padding(self, instr_blocks, adj_matrix, actual_nodes):
		"""Subgraph selection algorithm based on node importance (optimized process)"""
		# Select key nodes
		selected_indices = self.select_key_nodes(adj_matrix, actual_nodes, self.max_nodes)
		n_selected = len(selected_indices)
		
		# Reconstruct instruction block sequence
		processed_blocks = []
		for idx in selected_indices:
			block = instr_blocks[idx]
			processed_block = tokenize_and_pad(
				text=block,
				tokenizer=self.tokenizer,
				seq_len=self.seq_len
			)
			processed_blocks.append(torch.tensor(processed_block))
		
		# Pad insufficient blocks
		for _ in range(self.max_nodes - n_selected):
			processed_blocks.append(
				torch.tensor([self.tokenizer.pad_token_id] * self.seq_len)
			)
		input_ids = torch.stack(processed_blocks, dim=0)
		
		# Process adjacency matrix (only need to reconstruct if actual node count exceeds max_nodes)
		if actual_nodes <= self.max_nodes:
			# Not enough nodes, just pad
			adj_array = adj_matrix.toarray()
			padded_adj = np.zeros((self.max_nodes, self.max_nodes))
			padded_adj[:actual_nodes, :actual_nodes] = adj_array
			return input_ids, torch.tensor(padded_adj, dtype=torch.float32)
		
		# Reconstruct adjacency matrix
		row, col, data = adj_matrix.row, adj_matrix.col, adj_matrix.data
		index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
		
		# Filter edges between selected nodes
		mask = np.isin(row, selected_indices) & np.isin(col, selected_indices)
		selected_row = row[mask]
		selected_col = col[mask]
		selected_data = data[mask] if data is not None else np.ones(selected_row.shape[0])
		
		# Fix: Handle the case with no edges
		if len(selected_row) == 0:
			# No edges, create an all-zero adjacency matrix
			new_adj = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
		else:
			# Remap node indices - use list comprehension instead of np.vectorize
			new_row = [index_map[r] for r in selected_row]
			new_col = [index_map[c] for c in selected_col]
			
			# Create new sparse adjacency matrix and convert to dense format
			new_adj = coo_matrix(
			(selected_data, (new_row, new_col)),
			shape=(self.max_nodes, self.max_nodes)
			).toarray()
		
		return input_ids, torch.tensor(new_adj, dtype=torch.float32)

class SparseFunctionPairDataset(FunctionPairDataset):
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
		
		# Process the first function (sparse format)
		a_input_ids, a_adj_indices, a_adj_values = self.process_function(a_data)
		
		# Process the second function (sparse format)
		t_input_ids, t_adj_indices, t_adj_values = self.process_function(t_data)
		
		label = torch.tensor(row["label"], dtype=torch.float32)
		
		return (
			a_input_ids, 
			a_adj_indices, 
			a_adj_values,
			t_input_ids, 
			t_adj_indices, 
			t_adj_values,
			label
		)
	def process_function(self, func_data):
		instr_blocks = func_data["instruction_blocks"]
		adj_data = func_data["adjacency_matrix"]
		n_blocks = adj_data["shape"][0]
		
		# Create sparse adjacency matrix
		adj = coo_matrix(
			(adj_data["data"], (adj_data["row"], adj_data["col"])),
			shape=adj_data["shape"]
		)
		
		return self._process_with_sparse(instr_blocks, adj, n_blocks)

	def _process_with_sparse(self, instr_blocks, adj_matrix, actual_nodes):
		"""Process function and return sparse representation"""
		processed_blocks = []
		
		# 1. Process instruction blocks
		for i, block in enumerate(instr_blocks):
			if i >= self.max_nodes:
				break
			processed_block = tokenize_and_pad(block, self.tokenizer, self.seq_len)
			processed_blocks.append(torch.tensor(processed_block))
		
		if len(processed_blocks) < self.max_nodes:
			for _ in range(self.max_nodes - len(processed_blocks)):
				processed_blocks.append(
					torch.tensor([self.tokenizer.pad_token_id] * self.seq_len)
				)
		
		input_ids = torch.stack(processed_blocks, dim=0)
		
		# 2. Process adjacency matrix - sparse representation
		adj_matrix = adj_matrix.tocoo()
		rows = adj_matrix.row
		cols = adj_matrix.col
		
		# Filter out indices exceeding max_nodes
		valid_mask = (rows < self.max_nodes) & (cols < self.max_nodes)
		rows = rows[valid_mask]
		cols = cols[valid_mask]
		data = adj_matrix.data[valid_mask]
		
		# Combine indices into a (2, num_edges) numpy array
		indices = np.vstack((rows, cols))
		
		# Convert to tensors
		indices = torch.tensor(indices, dtype=torch.long)
		values = torch.tensor(data, dtype=torch.float32)
		
		return input_ids, indices, values