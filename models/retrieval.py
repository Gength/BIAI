"""Shared full-pool retrieval protocol for paper Task 1."""
import pickle

import torch
import torch.nn.functional as F


def binary_target(file_name):
    """Executable/library identity after the Dataset-1 metadata prefix."""
    if "_" not in file_name:
        raise ValueError(f"unexpected Dataset-1 binary name: {file_name!r}")
    return file_name.split("_", 1)[1]


def source_key(function_key):
    """Ground-truth identity: symbol within the same executable/library."""
    return function_key[0], binary_target(function_key[5])


def load_function_keys(path):
    with open(path, "rb") as handle:
        return [tuple(key) for key in pickle.load(handle)]


def build_retrieval_sets(anchor_keys, candidate_keys, opt_level=None):
    """Filter pools and attach every relevant candidate to each anchor."""
    anchors = [tuple(k) for k in anchor_keys
               if (not opt_level or k[3] == opt_level) and k[4] == "x64"]
    anchors = list(dict.fromkeys(anchors))
    candidates = [tuple(k) for k in candidate_keys
                  if (not opt_level or k[3] == opt_level) and k[4] == "arm64"]
    candidates = list(dict.fromkeys(candidates))
    positions_by_source = {}
    for position, key in enumerate(candidates):
        positions_by_source.setdefault(source_key(key), []).append(position)
    retrieval = [
        (anchor, positions_by_source[source_key(anchor)])
        for anchor in anchors
        if source_key(anchor) in positions_by_source
    ]
    return retrieval, candidates


def encode_keys(model, dataset, device, keys):
    """Encode function keys one native-size CFG at a time."""
    embeddings = []
    for key in keys:
        index = dataset.mapping[key]
        ids, adj = dataset.process_function(dataset.dataset[index])
        with torch.no_grad():
            embedding = model([ids.to(device)], [adj]).squeeze(0).cpu()
        embeddings.append(embedding)
    if not embeddings:
        raise RuntimeError("cannot encode an empty function pool")
    return torch.stack(embeddings, dim=0)


def evaluate_retrieval(model, dataset, device, anchor_keys, candidate_keys,
                       opt_level=None, progress=None):
    """Return paper-style full-pool MRR10 and Rank1."""
    retrieval, candidates = build_retrieval_sets(
        anchor_keys, candidate_keys, opt_level)
    if not retrieval:
        raise RuntimeError("no eligible cross-platform anchors were found")
    candidate_embeddings = encode_keys(model, dataset, device, candidates)
    iterator = progress(retrieval) if progress is not None else retrieval
    reciprocal_ranks = []
    rank1 = []
    for anchor, true_positions in iterator:
        anchor_embedding = encode_keys(model, dataset, device, [anchor])
        similarities = F.cosine_similarity(
            anchor_embedding, candidate_embeddings, dim=1)
        best_true_score = similarities[true_positions].max()
        rank = int((similarities > best_true_score).sum().item()) + 1
        reciprocal_ranks.append(1.0 / rank if rank <= 10 else 0.0)
        rank1.append(1.0 if rank == 1 else 0.0)
    return {
        "mrr10": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "rank1": sum(rank1) / len(rank1),
        "anchors": len(retrieval),
        "candidates": len(candidates),
    }
