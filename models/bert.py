import torch
import math
import torch.nn.functional as F
import torch.nn as nn



class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Input: 
            x: token indices [batch_size, seq_len]
        Output: 
            positional embeddings [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        return self.pos_embedding(positions)

class FeedForward(torch.nn.Module):
	"Implements FFN equation."

	def __init__(self, d_model, middle_dim=256, dropout=0.1):
		super(FeedForward, self).__init__()
		
		self.fc1 = torch.nn.Linear(d_model, middle_dim)
		self.fc2 = torch.nn.Linear(middle_dim, d_model)
		self.dropout = torch.nn.Dropout(dropout)
		self.activation = torch.nn.GELU()

	def forward(self, x):
		"""
		Input: 
			x: embeddings [batch_size, seq_len, d_model]
		Output: 
			transformed embeddings [batch_size, seq_len, d_model]
		"""
		out = self.activation(self.fc1(x))
		out = self.fc2(self.dropout(out))
		return out

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1, d_k=8):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_k

    def forward(self, query, key, value, mask=None):
        """
        Inputs:
            query: [batch_size, heads, seq_len, d_k]
            key: [batch_size, heads, seq_len, d_k]
            value: [batch_size, heads, seq_len, d_k]
            mask: optional [batch_size, 1, 1, seq_len]
        Output: 
            attention output [batch_size, heads, seq_len, d_k]
            attention weights [batch_size, heads, seq_len, seq_len]
        """
        assert(self.d_k == query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, value), attn_weights

class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.d_model = d_model

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout, self.d_k)
    
    def forward(self, query, key, value, mask=None):
        """
        Inputs:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: optional [batch_size, 1, 1, seq_len]
        Output: 
            attention output [batch_size, seq_len, d_model]
        """
        B, L, _ = query.size()

        Q = self.q_linear(query).view(B, L, self.heads, self.d_k).transpose(1, 2)  # [B, h, L, d_k]
        K = self.k_linear(key).view(B, L, self.heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(B, L, self.heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.expand(-1, self.heads, -1, -1)


        attn_output, _ = self.attention(Q, K, V, mask)  # [B, h, L, d_k]

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)  # [B, L, d_model]
        return self.out_proj(attn_output)
    
class MaskedLanguageModel(torch.nn.Module):
	"""
	predicting origin token from masked input sequence
	n-class classification problem, n-class = vocab_size
	"""

	def __init__(self, hidden, vocab_size):
		"""
		:param hidden: output size of BERT model
		:param vocab_size: total vocab size
		"""
		super().__init__()
		self.linear = torch.nn.Linear(hidden, vocab_size)
		self.softmax = torch.nn.LogSoftmax(dim=-1)

	def forward(self, x):
		"""
		Input: 
			x: hidden states [batch_size, seq_len, d_model]
		Output: 
			log probabilities [batch_size, seq_len, vocab_size]
		"""
		return self.softmax(self.linear(x))

class EncoderLayer(torch.nn.Module):
	def __init__(
		self, 
		d_model=768,
		heads=12, 
		feed_forward_hidden=768 * 4, 
		dropout=0.1
		):
		super(EncoderLayer, self).__init__()
		self.layernorm = torch.nn.LayerNorm(d_model)
		self.self_multihead = MultiHeadedAttention(heads, d_model)
		self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
		self.dropout = torch.nn.Dropout(dropout)

	def forward(self, embeddings, mask):
		"""
		Inputs:
			embeddings: [batch_size, seq_len, d_model]
			mask: [batch_size, 1, 1, seq_len]
		Output: 
			encoded representations [batch_size, seq_len, d_model]
		"""
		interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
		# residual layer
		interacted = self.layernorm(interacted + embeddings)
		# bottleneck
		feed_forward_out = self.dropout(self.feed_forward(interacted))
		encoded = self.layernorm(feed_forward_out + interacted)
		return encoded


class BERTEmbedding(torch.nn.Module):
	"""
	BERT Embedding which is consisted with under features
		1. TokenEmbedding : normal embedding matrix
		2. PositionalEmbedding : adding positional information using sin, cos
		2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
		sum of all these features are output of BERTEmbedding
	"""

	def __init__(self, vocab_size, embed_size=128, seq_len=128, dropout=0.1):
		"""
		:param vocab_size: total vocab size
		:param embed_size: embedding size of token embedding
		:param dropout: dropout rate
		"""

		super().__init__()
		self.embed_size = embed_size
		# (m, seq_len) --> (m, seq_len, embed_size)
		# padding_idx is not updated during training, remains as fixed pad (0)
		self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
		self.dropout = torch.nn.Dropout(p=dropout)
	   
	def forward(self, sequence):
		"""
		Input: 
			sequence: token indices [batch_size, seq_len]
		Output: 
			combined embeddings [batch_size, seq_len, embed_size]
		"""
		token_embedding = self.token(sequence)
		position_embedding = self.position(sequence)
		x = token_embedding + position_embedding
		return self.dropout(x)

class BERT_Block(torch.nn.Module):
	"""
	BERT model : Bidirectional Encoder Representations from Transformers.
	"""

	def __init__(self, vocab_size, d_model=128, n_layers=12, heads=8, seq_len=128,dropout=0.1,  device="cuda"):
		"""
		:param vocab_size: vocab_size of total words
		:param hidden: BERT model hidden size
		:param n_layers: numbers of Transformer blocks(layers)
		:param attn_heads: number of attention heads
		:param dropout: dropout rate
		"""

		super().__init__()
		self.d_model = d_model
		self.n_layers = n_layers
		self.heads = heads
		self.seq_len = seq_len

		self.feed_forward_hidden = d_model * 2

		# embedding for BERT, sum of positional, segment, token embeddings
		self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model, seq_len=seq_len)

		# multi-layers transformer blocks, deep network
		self.encoder_blocks = torch.nn.ModuleList(
			[EncoderLayer(d_model, heads, d_model * 2, dropout) for _ in range(n_layers)])
		self.mask_lm = MaskedLanguageModel(self.d_model, vocab_size)
		self.device = device

	def forward(self, x):
		"""
		Input: 
			x: token indices [batch_size, seq_len]
		Output: 
			MLM logits [batch_size, seq_len, vocab_size]
		"""
		# attention masking for padded token
		# (batch_size, 1, seq_len, seq_len)
		# mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
		x = x.to(self.device)
		mask = (x > 0).unsqueeze(1).unsqueeze(2)

		# embedding the indexed sequence to sequence of vectors
		x = self.embedding(x)

		# running over multiple transformer blocks
		for encoder in self.encoder_blocks:
			x = encoder.forward(x, mask)
		x = self.mask_lm(x)
		return x

	def encode(self, input):
		"""
		Input: 
			input: token indices [batch_size, seq_len]
		Output: 
			pooled representation [batch_size, d_model]
		"""
		input = input.to(self.device)
		mask = (input > 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
		x = self.embedding(input)
		for encoder in self.encoder_blocks:
			x = encoder(x, mask)
		attention_mask = (input > 0).float().unsqueeze(-1)  # (B,L,1)
		summed = torch.sum(x * attention_mask, dim=1)
		return summed

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=12, 
                 heads=8, seq_len=128, dropout=0.1, device="cuda"):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        
        # Shared BERT encoder
        self.bert = BERT_Block(vocab_size, d_model, n_layers, heads, seq_len, dropout, device)
        
        # MLM task head
        self.mlm_head = MaskedLanguageModel(d_model, vocab_size)

    def forward(self, input_dict):
        """
        Unified interface for MLM tasks
        Input: 
            input_dict: dictionary with task-specific keys (assumed to be on correct device)
        Output: 
            (loss, logits) tuple
        """
        task_type = input_dict['task_type']
        
        if task_type == 'mlm':
            return self.forward_mlm(input_dict['input_ids'], input_dict['labels'])
        else:
            raise ValueError(f"Unknown Task Type: {task_type}")

    def forward_mlm(self, input_ids, labels):
        """
        Inputs (assumed to be on correct device):
            input_ids: token indices [batch_size, seq_len]
            labels: target token IDs [batch_size, seq_len]
        Output: 
            (loss, logits) where logits [batch_size, seq_len, vocab_size]
        """
        log_probs = self.bert(input_ids)
        loss = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), labels.view(-1), ignore_index=0)
        return loss, log_probs

    def encode(self, input_ids):
        """
        Input (assumed to be on correct device):
            input_ids: token indices [batch_size, seq_len]
        Output: 
            block embedding [batch_size, d_model]
        """
        return self.bert.encode(input_ids)

class ANPHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, vec):
        """
        Inputs:
            vec: block embedding [batch_size, d_model]
        Output: 
            adjacency logits [batch_size, 2]
        """
        return self.classifier(vec)
	
class BERT2(BERT):
    def __init__(self, vocab_size, d_model=128, n_layers=12, 
                 heads=8, seq_len=128, dropout=0.1, device="cuda"):
        super().__init__(vocab_size, d_model, n_layers, heads, seq_len, dropout, device)
        
        # ANP task head
        self.anp_head = ANPHead(d_model)

    def forward(self, input_dict):
        """
        Unified interface for MLM and ANP tasks
        Input: 
            input_dict: dictionary with task-specific keys (assumed to be on correct device)
        Output: 
            (loss, logits) tuple
        """
        task_type = input_dict['task_type']
        
        if task_type == 'mlm':
            return self.forward_mlm(input_dict['input_ids'], input_dict['labels'])
        elif task_type == 'anp':
            return self.forward_anp(
                input_dict['input_ids'], 
                input_dict['labels']
            )
        else:
            raise ValueError(f"Unknown Task Type: {task_type}")

    def forward_anp(self, input, labels):
        """
        judge whether two blocks are adjacent in a graph
        Inputs (assumed to be on correct device):
            input: concatenated 2 block tokens [batch_size, seq_len]
            labels: adjacency labels [batch_size]
        Output: 
            (loss, logits) where logits [batch_size, 2]
        """
        vec = self.encode(input)
        logits = self.anp_head(vec)
        loss = F.cross_entropy(logits, labels)
        return loss, logits

class BIGHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            # Binary classification: same graph / different graph
            nn.Linear(hidden_dim, 2)  
        )

    def forward(self, vec_a, vec_b):
        """
        Inputs:
            vec_a: block embedding [batch_size, d_model]
            vec_b: block embedding [batch_size, d_model]
        Output: 
            same-graph logits [batch_size, 2]
        """
        x = torch.cat([vec_a, vec_b], dim=1)
        return self.classifier(x)

class GraphClassificationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Multi-class: different platforms/optimization options
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, vec):
        """
        Input: 
            vec: block embedding [batch_size, d_model]
        Output: 
            class logits [batch_size, num_classes]
        """
        return self.classifier(vec)

class BERT4(BERT2):
    def __init__(self, vocab_size, num_classes, d_model=128, n_layers=12, 
                 heads=8, seq_len=128, dropout=0.1, device="cuda"):
        super().__init__(vocab_size, d_model, n_layers, heads, seq_len, dropout, device)
        
        # Add two new task heads
        self.big_head = BIGHead(d_model)  # Task: whether blocks are in the same graph
        self.gc_head = GraphClassificationHead(d_model, num_classes)  # Graph classification task

    def forward(self, input_dict):
        """
        Unified interface for 4 tasks (MLM, ANP, BIG, GC)
        Input: 
            input_dict: dictionary with task-specific keys (assumed to be on correct device)
        Output: 
            (loss, logits) tuple
        """
        task_type = input_dict['task_type']
        
        if task_type in ['mlm', 'anp']:
            return super().forward(input_dict)
        elif task_type == 'big':
            return self.forward_big(
                input_dict['input_a'], 
                input_dict['input_b'], 
                input_dict['labels']
            )
        elif task_type == 'gc':
            return self.forward_gc(
                input_dict['input_ids'], 
                input_dict['labels']
            )
        else:
            raise ValueError(f"Unknown Task Type: {task_type}")

    def forward_big(self, input_a, input_b, labels):
        """
        Inputs (assumed to be on correct device):
            input_a: block A tokens [batch_size, seq_len]
            input_b: block B tokens [batch_size, seq_len]
            labels: same-graph labels [batch_size]
        Output: 
            (loss, logits) where logits [batch_size, 2]
        """
        vec_a = self.encode(input_a)
        vec_b = self.encode(input_b)
        logits = self.big_head(vec_a, vec_b)
        loss = F.cross_entropy(logits, labels)
        return loss, logits

    def forward_gc(self, input_ids, labels):
        """
        Inputs (assumed to be on correct device):
            input_ids: token indices [batch_size, seq_len]
            labels: graph class labels [batch_size]
        Output: 
            (loss, logits) where logits [batch_size, num_classes]
        """
        vec = self.encode(input_ids)
        logits = self.gc_head(vec)
        loss = F.cross_entropy(logits, labels)
        return loss, logits