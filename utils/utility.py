from models.tokenizer import AsmTokenizer


def tokenize_and_pad(text: list, tokenizer: AsmTokenizer, seq_len: int) -> list:
    """
    Tokenizes and adds \<CLS\>, \<SEP\>, and padding tokens to a given text.

    Args:
        tokenizer: The tokenizer to use for encoding.
        text (str): The text to tokenize.
        seq_len (int): The desired sequence length.

    Returns:
        list: A list of token IDs padded to the specified sequence length.
    """
    ids = tokenizer.encode(text)
    return add_cls_sep_pad(ids, tokenizer, seq_len)


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
