"""BERT pre-training models for CFG blocks, built on HuggingFace transformers.

Replaces the hand-written Transformer with `transformers.BertModel` while
keeping the four pre-training tasks from the Order-Matters paper:

- MLM (token-level): masked language model on the block token sequence.
- ANP (block-level):  predict whether two blocks are adjacent in the CFG.
- BIG (graph-level): predict whether two blocks belong to the same CFG.
- GC  (graph-level): predict the (opt, arch) class of the block's function.

Model size follows the paper: hidden 128, 12 layers, 8 heads, FFN 256,
max seq len 128 (~1.6M params).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


def build_bert_config(vocab_size, seq_len=128, d_model=128, n_layers=12,
                      heads=8, ff_dim=256, pad_token_id=0):
    """BERT config aligned with the Order-Matters paper."""
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=d_model,
        num_hidden_layers=n_layers,
        num_attention_heads=heads,
        intermediate_size=ff_dim,
        max_position_embeddings=seq_len,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=2,
        pad_token_id=pad_token_id,
    )
    # Use PyTorch's fused SDPA kernel (flash / memory-efficient) instead of
    # eager attention. Free ~3% speedup; flash-attn itself is not worth
    # installing for seq_len=128 / hidden=128 (attention is not the bottleneck).
    config._attn_implementation = "sdpa"
    return config


class BERTForPretraining(nn.Module):
    """HuggingFace BertModel + task heads (MLM / ANP / BIG / GC).

    Args:
        config: BertConfig for the underlying BERT encoder.
        num_gc_classes: number of classes for the GC task.
        tasks: tuple of enabled tasks, subset of ("mlm", "anp", "big", "gc").
    """

    def __init__(self, config, num_gc_classes=10, tasks=("mlm", "anp", "big", "gc")):
        super().__init__()
        self.config = config
        self.tasks = tuple(tasks)
        self.num_gc_classes = num_gc_classes

        self.bert = BertModel(config, add_pooling_layer=True)

        # MLM head (token-level), with weights tied to the token embeddings.
        self.mlm_head = BertOnlyMLMHead(config)
        self.mlm_head.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

        # ANP / BIG heads (block-pair classification, 2 classes).
        self.anp_head = nn.Linear(config.hidden_size, 2)
        self.big_head = nn.Linear(config.hidden_size, 2)
        # GC head (graph classification).
        self.gc_head = nn.Linear(config.hidden_size, num_gc_classes)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                mlm_labels=None, anp_labels=None, big_labels=None, gc_labels=None,
                return_logits=False):
        """Run the BERT encoder and compute the requested task losses.

        Returns a dict with keys: `losses` (dict of enabled-task losses),
        `logits` (dict of computed logits), `pooled` ([CLS] block embeddings).
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state  # [B, L, H]
        pooled = outputs.pooler_output               # [B, H]

        logits = {}
        if "mlm" in self.tasks and (return_logits or mlm_labels is not None):
            logits["mlm"] = self.mlm_head(sequence_output)
        if "anp" in self.tasks and (return_logits or anp_labels is not None):
            logits["anp"] = self.anp_head(pooled)
        if "big" in self.tasks and (return_logits or big_labels is not None):
            logits["big"] = self.big_head(pooled)
        if "gc" in self.tasks and (return_logits or gc_labels is not None):
            logits["gc"] = self.gc_head(pooled)

        losses = {}
        if "mlm" in self.tasks and mlm_labels is not None:
            losses["mlm"] = F.cross_entropy(
                logits["mlm"].view(-1, self.config.vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100,
            )
        if "anp" in self.tasks and anp_labels is not None:
            losses["anp"] = F.cross_entropy(logits["anp"], anp_labels)
        if "big" in self.tasks and big_labels is not None:
            losses["big"] = F.cross_entropy(logits["big"], big_labels)
        if "gc" in self.tasks and gc_labels is not None:
            losses["gc"] = F.cross_entropy(logits["gc"], gc_labels)

        return {"losses": losses, "logits": logits, "pooled": pooled}

    # ------------------------------------------------------------------ #
    # Block embedding extraction (used by CFGFusionModel at fine-tune time)
    # ------------------------------------------------------------------ #
    def encode_block_embeddings(self, input_ids, attention_mask=None,
                                token_type_ids=None):
        """Encode a batch of block sequences into [CLS] block embeddings.

        Inputs: [B, L] tensors (B = batch * num_nodes).
        Output: [B, hidden_size] block embeddings (pooler output).
        """
        pooled = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).pooler_output
        return pooled

    # ------------------------------------------------------------------ #
    # Checkpoint helpers
    # ------------------------------------------------------------------ #
    def save_pretrained(self, save_path):
        """Save model weights + config to `save_path` (a directory)."""
        os.makedirs(save_path, exist_ok=True)
        self.config.save_pretrained(save_path)
        torch.save(
            {
                "model": self.state_dict(),
                "num_gc_classes": self.num_gc_classes,
                "tasks": list(self.tasks),
            },
            os.path.join(save_path, "pytorch_model.bin"),
        )

    @classmethod
    def from_pretrained(cls, save_path, device=None, strict=True):
        """Load a model saved by `save_pretrained`."""
        config = BertConfig.from_pretrained(save_path)
        checkpoint = torch.load(
            os.path.join(save_path, "pytorch_model.bin"),
            map_location=device or ("cpu" if not torch.cuda.is_available() else "cuda"),
            weights_only=True,
        )
        model = cls(
            config,
            num_gc_classes=checkpoint.get("num_gc_classes", 10),
            tasks=checkpoint.get("tasks", ("mlm", "anp", "big", "gc")),
        )
        model.load_state_dict(checkpoint["model"], strict=strict)
        return model
