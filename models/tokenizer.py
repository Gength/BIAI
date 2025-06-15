import os
import re
import string

class AsmTokenizer:
	def __init__(self, corpus=None, vocab_file=None, max_vocab_size=5000):
		self.max_vocab_size = max_vocab_size
		self.vocab = {"<PAD>": 0, "<CLS>": 1, "<SEP>": 2, "<MASK>": 3, "<UNK>": 4, "<const>": 5}
		self.rev_vocab = {v: k for k, v in self.vocab.items()}
		if vocab_file and os.path.exists(vocab_file):
			self.load_vocab(vocab_file)
		elif corpus:
			self.build_vocab(corpus)
			self.save_vocab(vocab_file)
					
	def build_vocab(self, corpus):
		idx = len(self.vocab)
		for line in corpus:
			tokens = self.tokenize(line)
			for tok in tokens:
				if tok not in self.vocab:
					if len(self.vocab) >= self.max_vocab_size:
						return
					self.vocab[tok] = idx
					self.rev_vocab[idx] = tok
					idx += 1

	def save_vocab(self, filepath):
		with open(filepath, "w") as f:
			for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
				f.write(f"{token}\n")
		print(f"Vocab saved to {filepath}")
	
	def load_vocab(self, filepath):
		with open(filepath, "r") as f:
			for i, line in enumerate(f):
				token = line.strip()
				if i >= self.max_vocab_size:
					break
				self.vocab[token] = i
				self.rev_vocab[i] = token
		print(f"Vocab loaded from {filepath}")

	def tokenize(self, text):
		# Use a more concise regular expression
		token_pattern = r"""
			\<[A-Za-z]+:[A-Za-z0-9_]+(?::\w+)?\> |  # <TYPE:value> or <TYPE:subtype:value>
			\<[A-Z]+\> |                            # All uppercase label <TARGET>
			\[[^\]]+\] |                            # Complete memory expression
			[A-Za-z0-9_]+:[A-Za-z0-9_]+ |           # category:opcode
			[\[\],:;*+./-] |                        # Single special character
			[\w]+                                   # Regular word characters
		"""
		tokens = re.findall(token_pattern, text, re.VERBOSE)
		tokens = [token.strip() for token in tokens if token.strip()]
		return self.expand_tokens(tokens)

	def expand_tokens(self, tokens):
		expanded = []
		for token in tokens:
			# Split opcode category
			# Handle memory expression
			if token.startswith('[') and token.endswith(']'):
				expanded.extend(self.expand_memory(token))
			elif ':' in token and not token.startswith('<'):
				parts = token.split(':')
				expanded.append(parts[0])  # category part
				expanded.append(':')       # add separator
				expanded.append(parts[1])  # opcode part
			else:
				expanded.append(token)
		return expanded

	def expand_memory(self, token):
		if token.startswith('[') and token.endswith(']'):
			# Decompose memory expression
			expanded = []
			inner = token[1:-1]
			if inner:
				expanded.append('[')
				# Split by + but keep the operator
				parts = re.split(r'(\+)', inner)
				expanded.extend([p for p in parts if p])
				expanded.append(']')
			else:
				expanded = token
		else:
			expanded = token
		return expanded

	def encode(self, text):
		tokens = self.tokenize(text)
		return [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in tokens]

	def decode(self, token_ids):
		return " ".join([self.rev_vocab.get(tok_id, "<UNK>") for tok_id in token_ids])
