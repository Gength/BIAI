"""Asm tokenizer for normalized assembly code, built on HuggingFace PreTrainedTokenizer.

Keeps the regex-based tokenization of normalized assembly instructions
(see pipeline.md), but exposes the full HuggingFace tokenizer API:
`tokenizer(text, ...)`, `from_pretrained`, `save_pretrained`, etc.
"""
import os
import re
from transformers import PreTrainedTokenizer

# Initial vocabulary (order matters: ids must stay stable across runs).
BASE_VOCAB = {"<PAD>": 0, "<CLS>": 1, "<SEP>": 2, "<MASK>": 3, "<UNK>": 4, "<const>": 5}

SPECIAL_TOKEN_ATTRS = [
    ("pad_token", "<PAD>"),
    ("cls_token", "<CLS>"),
    ("sep_token", "<SEP>"),
    ("mask_token", "<MASK>"),
    ("unk_token", "<UNK>"),
]


class AsmTokenizer(PreTrainedTokenizer):
    """Tokenizer for normalized assembly code (HuggingFace-compatible)."""

    vocab_files_names = {"vocab_file": "vocab.txt"}

    def __init__(self, vocab_file=None, corpus=None, max_vocab_size=5000, **kwargs):
        # 1. Build vocab BEFORE super().__init__() (the parent reads it via
        #    `get_vocab()` when registering special tokens; tokens that are
        #    already in the base vocab are automatically skipped).
        if vocab_file and os.path.exists(vocab_file):
            self.vocab = {}
            with open(vocab_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    token = line.strip()
                    if token:
                        self.vocab[token] = i
        else:
            self.vocab = dict(BASE_VOCAB)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.max_vocab_size = max_vocab_size

        # 2. Initialize the parent.
        super().__init__(**kwargs)

        # 3. Register special token attributes.
        for attr, token in SPECIAL_TOKEN_ATTRS:
            setattr(self, attr, token)

        # 4. Make sure token_type_ids (segment ids) are generated.
        self.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

        # 5. Optionally build the vocab from a corpus and persist it.
        if corpus is not None and not (vocab_file and os.path.exists(vocab_file)):
            self.build_vocab(corpus)
            if vocab_file:
                os.makedirs(os.path.dirname(vocab_file) or ".", exist_ok=True)
                self.save_vocabulary(os.path.dirname(vocab_file) or ".")

    # ------------------------------------------------------------------ #
    # Vocab construction / persistence
    # ------------------------------------------------------------------ #
    def build_vocab(self, corpus, max_vocab_size=None):
        """Add tokens seen in `corpus` (iterable of lines / blocks) to the vocab."""
        if max_vocab_size is None:
            max_vocab_size = self.max_vocab_size
        idx = len(self.vocab)
        for line in corpus:
            for tok in self._tokenize(line):
                if tok not in self.vocab:
                    if len(self.vocab) >= max_vocab_size:
                        return
                    self.vocab[tok] = idx
                    self.ids_to_tokens[idx] = tok
                    idx += 1

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save the vocab to `<save_directory>/vocab.txt` (HF convention)."""
        path = os.path.join(save_directory, "vocab.txt")
        with open(path, "w", encoding="utf-8") as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\n")
        return (path,)

    def save_vocab(self, filepath):
        """Save the vocab to an explicit file path (legacy helper)."""
        self.save_vocabulary(os.path.dirname(filepath) or ".")
        saved = os.path.join(os.path.dirname(filepath) or ".", "vocab.txt")
        if os.path.abspath(saved) != os.path.abspath(filepath):
            os.replace(saved, filepath)
        print(f"Vocab saved to {filepath}")

    # ------------------------------------------------------------------ #
    # HF tokenizer interface
    # ------------------------------------------------------------------ #
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """BERT-style: <CLS> seq0 <SEP> [seq1 <SEP>]."""
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return ([self.cls_token_id] + token_ids_0 + [self.sep_token_id]
                + token_ids_1 + [self.sep_token_id])

    def _tokenize(self, text):
        token_pattern = r"""
            \<[A-Za-z]+:[A-Za-z0-9_]+(?::\w+)?\> |  # <TYPE:value> or <TYPE:subtype:value>
            \<[A-Z]+\> |                            # All uppercase label <TARGET>
            \[[^\]]+\] |                            # Complete memory expression
            [A-Za-z0-9_]+:[A-Za-z0-9_]+ |           # category:opcode
            [\[\],:;*+./-] |                        # Single special character
            [\w]+                                   # Regular word characters
        """
        tokens = re.findall(token_pattern, text, re.VERBOSE)
        tokens = [t.strip() for t in tokens if t.strip()]
        return self.expand_tokens(tokens)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab["<UNK>"])

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, "<UNK>")

    def get_vocab(self):
        return dict(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ------------------------------------------------------------------ #
    # Assembly-specific helpers
    # ------------------------------------------------------------------ #
    def expand_tokens(self, tokens):
        expanded = []
        for token in tokens:
            # Handle memory expression
            if token.startswith("[") and token.endswith("]"):
                expanded.extend(self.expand_memory(token))
            elif ":" in token and not token.startswith("<"):
                parts = token.split(":")
                expanded.append(parts[0])  # category part
                expanded.append(":")       # add separator
                expanded.append(parts[1])  # opcode part
            else:
                expanded.append(token)
        return expanded

    def expand_memory(self, token):
        if token.startswith("[") and token.endswith("]"):
            # Decompose memory expression
            expanded = []
            inner = token[1:-1]
            if inner:
                expanded.append("[")
                # Split by + but keep the operator
                parts = re.split(r"(\+)", inner)
                expanded.extend([p for p in parts if p])
                expanded.append("]")
            else:
                return token
            return expanded
        return token

    def encode_block(self, block, seq_len=128, return_tensors=None):
        """Encode a single basic block into `<CLS> block <SEP>` (BERT format).

        Returns a BatchEncoding with `input_ids`, `token_type_ids` and
        `attention_mask` padded/truncated to `seq_len`.
        """
        return self(
            block,
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
        )

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token ids back to a token string (HF-compatible signature)."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        tokens = [self._convert_id_to_token(int(t)) for t in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.all_special_tokens]
        return " ".join(tokens)
