from models.tokenizer import AsmTokenizer
import random
def random_mask(ids, tokenizer: AsmTokenizer):
    output = []
    labels = []
    for token_id in ids:
        if token_id == tokenizer.sep_token_id or token_id == tokenizer.cls_token_id:
            output.append(token_id)
            labels.append(0)
        elif random.random() < 0.15:
            rand_val = random.random()
            if rand_val < 0.8:
                output.append(tokenizer.mask_token_id)  # 80% replaced with MASK
            elif rand_val < 0.9:
                output.append(random.choice(list(tokenizer.vocab.values())))  # 10% random token
            else:
                output.append(token_id)  # 10% keep original
            labels.append(token_id)
        else:
            output.append(token_id)
            labels.append(0)
    return output, labels

def tokenize_and_pad(text: list, tokenizer: AsmTokenizer, seq_len: int) -> list:
    """
    Tokenize and Pad tokens to a given text.

    Args:
        tokenizer: The tokenizer to use for encoding.
        text (str): The text to tokenize.
        seq_len (int): The desired sequence length.

    Returns:
        list: A list of token IDs padded to the specified sequence length.
    """
    ids = tokenizer.encode(text)
    return pad_sequence(ids, tokenizer, seq_len, tokenizer.pad_token_id)


def add_cls_sep_pad(ids: list, tokenizer: AsmTokenizer, seq_len: int) -> list:
    """
    Adds <CLS>, <SEP>, and padding tokens to a list of token IDs.

    Args:
        tokenizer: The tokenizer to use for adding special tokens.
        ids (list): The list of token IDs to modify.
        seq_len (int): The desired sequence length.

    Returns:
        list: A list of token IDs with <CLS>, <SEP>, and padding added.
    """
    ids = ids[: seq_len - 2]  # Truncate to max length
    pad_len = seq_len - len(ids) - 2  # Subtract <CLS> and <SEP>
    return (
        [tokenizer.cls_token_id]
        + ids
        + [tokenizer.sep_token_id]
        + [tokenizer.pad_token_id] * pad_len
    )

def pad_sequence(ids: list, seq_len: int, pad_id) -> list:
    """
    Pad token ids to a given length.

    Args:
        ids (list): The list of token IDs to modify.
        seq_len (int): The desired sequence length.
        pad_id: The padding token ID.

    Returns:
        list: A list of token IDs with padding added.
    """
    pad_len = seq_len - len(ids)
    return (ids+ [pad_id] * pad_len
    )