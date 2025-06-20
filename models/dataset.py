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
from sklearn.cluster import SpectralClustering
import warnings
from sklearn.exceptions import ConvergenceWarning
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
	def __init__(self, csv_path, jsonl_path, mapping_path, tokenizer:AsmTokenizer, seq_len=128, max_blocks=50):
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
		self.seq_len = seq_len
		self.max_blocks = max_blocks

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
		"""
		Process a single function for model input
		Args:
			func_data: Dictionary containing function data
		Returns:
			input_ids: Tokenized instruction blocks (tensor)
			adj: Processed adjacency matrix (tensor)
		"""
		instr_blocks = func_data["instruction_blocks"]
		adj_data = func_data["adjacency_matrix"]
		n_blocks = adj_data["shape"][0]
		
		# Create sparse adjacency matrix
		original_adj = coo_matrix(
			(adj_data["data"], (adj_data["row"], adj_data["col"])),
			shape=adj_data["shape"]
		)
		
		# Handle different graph sizes
		if n_blocks <= self.max_blocks:
			return self._process_small_graph(instr_blocks, original_adj)
		elif n_blocks <= 1000:  # Medium-sized graph
			return self._coarsen_with_clustering(instr_blocks, original_adj)
		else:  # Very large graph
			return self._process_huge_graph(instr_blocks, original_adj)

	def _process_small_graph(self, instr_blocks, adj_matrix):
		"""
		Process graphs smaller than max_blocks with padding
		Args:
			instr_blocks: List of instruction strings
			adj_matrix: Sparse adjacency matrix (coo_matrix)
		Returns:
			input_ids: Tokenized blocks with padding
			adj: Padded adjacency matrix
		"""
		processed_blocks = []
		for block in instr_blocks:
			processed_block = tokenize_and_pad(
				text=block,
				tokenizer=self.tokenizer,
				seq_len=self.seq_len
			)
			processed_blocks.append(torch.tensor(processed_block))
		
		# Pad blocks if needed
		if len(processed_blocks) < self.max_blocks:
			for _ in range(self.max_blocks - len(processed_blocks)):
				processed_blocks.append(
					torch.tensor([self.tokenizer.pad_token_id] * self.seq_len)
				)
		
		input_ids = torch.stack(processed_blocks, dim=0)
		
		# Convert to dense array for padding
		adj_array = adj_matrix.toarray()
		if adj_array.shape[0] < self.max_blocks:
			padded_adj = np.zeros((self.max_blocks, self.max_blocks))
			padded_adj[:adj_array.shape[0], :adj_array.shape[1]] = adj_array
			adj_array = padded_adj
		
		return input_ids, torch.tensor(adj_array, dtype=torch.float32)

	def _coarsen_with_clustering(self, instr_blocks, adj_matrix):
		"""
		Coarsen medium-sized graphs using spectral clustering
		Args:
			instr_blocks: List of instruction strings
			adj_matrix: Sparse adjacency matrix (coo_matrix)
		Returns:
			input_ids: Tokenized coarsened blocks
			adj: Coarsened adjacency matrix
		"""
		# Convert to dense array for clustering
		adj_array = adj_matrix.toarray()
		n_blocks = adj_array.shape[0]
		
		if adj_array.dtype != np.float32 and adj_array.dtype != np.float64:
			adj_array = adj_array.astype(np.float32)

		# Create symmetric version for clustering
		symmetric_adj = adj_array + adj_array.T
		
		# Apply spectral clustering
		cluster_labels = self._spectral_clustering(symmetric_adj, self.max_blocks)
		
		# Create new adjacency matrix between supernodes
		new_adj = np.zeros((self.max_blocks, self.max_blocks), dtype=np.float32)
		for i in range(n_blocks):
			for j in range(n_blocks):
				if adj_array[i, j] > 0:  # Consider existing edges
					cluster_i = cluster_labels[i]
					cluster_j = cluster_labels[j]
					new_adj[cluster_i, cluster_j] += adj_array[i, j]
		
		# Normalize by cluster size to preserve density
		cluster_sizes = np.bincount(cluster_labels, minlength=self.max_blocks)
		with np.errstate(divide='ignore', invalid='ignore'):
			normalization = np.outer(cluster_sizes, cluster_sizes)
			new_adj = np.divide(new_adj, normalization, where=normalization>0)
		
		# Merge instruction blocks within clusters
		merged_blocks = []
		for cluster_id in range(self.max_blocks):
			block_indices = np.where(cluster_labels == cluster_id)[0]
			cluster_text = " <SEP> ".join(instr_blocks[i] for i in block_indices)
			tokenized = tokenize_and_pad(
				text=cluster_text,
				tokenizer=self.tokenizer,
				seq_len=self.seq_len
			)
			merged_blocks.append(torch.tensor(tokenized))
		
		input_ids = torch.stack(merged_blocks, dim=0)
		return input_ids, torch.tensor(new_adj, dtype=torch.float32)

	def _process_huge_graph(self, instr_blocks, adj_matrix):
		"""
		Process very large graphs (>1000 nodes) with truncation followed by clustering
		Args:
			instr_blocks: List of instruction strings
			adj_matrix: Sparse adjacency matrix (coo_matrix)
		Returns:
			input_ids: Tokenized blocks
			adj: Processed adjacency matrix
		"""
		# Determine truncation size (2x max_blocks but at least 100)
		trunc_size = max(2 * self.max_blocks, 100)
		n_blocks = adj_matrix.shape[0]
		
		# Step 1: Truncate graph to manageable size
		trunc_blocks = min(trunc_size, n_blocks)
		trunc_instr = instr_blocks[:trunc_blocks]
		
		# Fix: Correctly extract the subset of the COO matrix
		# Get the row and column indices to keep
		rows = adj_matrix.row
		cols = adj_matrix.col
		data = adj_matrix.data
		
		# Find edges among the first trunc_blocks nodes
		mask = (rows < trunc_blocks) & (cols < trunc_blocks)
		trunc_rows = rows[mask]
		trunc_cols = cols[mask]
		trunc_data = data[mask]
		
		# Create the truncated adjacency matrix
		trunc_adj = coo_matrix(
			(trunc_data, (trunc_rows, trunc_cols)),
			shape=(trunc_blocks, trunc_blocks),
		)
		trunc_adj_dense = trunc_adj.toarray()
		
		# Ensure float type
		if trunc_adj_dense.dtype != np.float32 and trunc_adj_dense.dtype != np.float64:
			trunc_adj_dense = trunc_adj_dense.astype(np.float32)
		
		# Step 2: Apply clustering on truncated graph
		symmetric_adj = trunc_adj_dense + trunc_adj_dense.T
		cluster_labels = self._spectral_clustering(symmetric_adj, self.max_blocks)
		
		# Create new adjacency matrix
		new_adj = np.zeros((self.max_blocks, self.max_blocks), dtype=np.float32)
		for i in range(trunc_blocks):
			for j in range(trunc_blocks):
				if trunc_adj_dense[i, j] > 0:
					cluster_i = cluster_labels[i]
					cluster_j = cluster_labels[j]
					new_adj[cluster_i, cluster_j] += trunc_adj_dense[i, j]
		
		# Normalize by cluster size
		cluster_sizes = np.bincount(cluster_labels, minlength=self.max_blocks)
		with np.errstate(divide='ignore', invalid='ignore'):
			normalization = np.outer(cluster_sizes, cluster_sizes)
			new_adj = np.divide(new_adj, normalization, where=normalization>0)
		
		# Merge instruction blocks within clusters
		merged_blocks = []
		for cluster_id in range(self.max_blocks):
			block_indices = np.where(cluster_labels == cluster_id)[0]
			cluster_text = " <SEP> ".join(trunc_instr[i] for i in block_indices)
			tokenized = tokenize_and_pad(
				text=cluster_text,
				tokenizer=self.tokenizer,
				seq_len=self.seq_len
			)
			merged_blocks.append(torch.tensor(tokenized))
		
		input_ids = torch.stack(merged_blocks, dim=0)
		return input_ids, torch.tensor(new_adj, dtype=torch.float32)

	def _spectral_clustering(self, adj_matrix, n_clusters):
		"""
		Perform spectral clustering on adjacency matrix
		Args:
			adj_matrix: Dense symmetric adjacency matrix
			n_clusters: Number of clusters to create
		Returns:
			Cluster labels for each node
		"""
		# Ensure labels are in the range [0, n_clusters-1]
		def normalize_labels(labels, n_clusters):
			unique = np.unique(labels)
			if len(unique) > n_clusters or np.max(labels) >= n_clusters:
				# Remap labels to the range 0~n_clusters-1
				_, labels = np.unique(labels, return_inverse=True)
				labels = labels % n_clusters  # Ensure no overflow
			return labels

		# Handle small matrices
		n_nodes = adj_matrix.shape[0]
		if n_nodes <= n_clusters:
			labels = np.arange(n_nodes)
			return normalize_labels(labels, n_clusters)
		
		# Ensure matrix is float type
		if adj_matrix.dtype != np.float32 and adj_matrix.dtype != np.float64:
			adj_matrix = adj_matrix.astype(np.float32)
		
		# Add stronger regularization to improve convergence
		regularization = 1e-4 * np.eye(adj_matrix.shape[0])
		adj_matrix += regularization
		
		# Apply spectral clustering
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", 
				message="Graph is not fully connected, spectral embedding may not work as expected.")
			warnings.filterwarnings("ignore", category=ConvergenceWarning)
			
			try:
				# For large graphs, use faster method
				if n_nodes > 500:
					clustering = SpectralClustering(
						n_clusters=n_clusters,
						affinity='precomputed',
						assign_labels='kmeans',
						random_state=42,
						n_init=10,
						eigen_tol=1e-3  # Lower precision requirement
					)
				else:
					clustering = SpectralClustering(
						n_clusters=n_clusters,
						affinity='precomputed',
						assign_labels='discretize',
						random_state=42,
						eigen_tol=1e-3  # Lower precision requirement
					)
				
				labels = clustering.fit_predict(adj_matrix)
				return normalize_labels(labels, n_clusters)
			
			except Exception as e:
				print(f"Spectral clustering failed: {e}, using fallback method")
				
				# Fallback 1: Connected components
				try:
					from scipy.sparse.csgraph import connected_components
					_, labels = connected_components(
						csgraph=adj_matrix, 
						directed=False, 
						return_labels=True
					)
					unique_labels = np.unique(labels)
					if len(unique_labels) < n_clusters:
						from sklearn.cluster import KMeans
						degrees = np.sum(adj_matrix, axis=1)
						kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
						labels = kmeans.fit_predict(degrees.reshape(-1, 1))
					return normalize_labels(labels, n_clusters)
				
				# Fallback 2: Degree-based clustering
				except:
					degrees = np.sum(adj_matrix, axis=1)
					sorted_indices = np.argsort(degrees)[::-1]
					labels = np.zeros(n_nodes, dtype=int)
					for i in range(n_nodes):
						labels[sorted_indices[i]] = i % n_clusters
					return normalize_labels(labels, n_clusters)