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


def encode_keys(model, dataset, device, keys, node_budget=4000):
    """Encode native-size CFGs in total-node-bounded BERT batches."""
    embeddings = []
    pending_ids, pending_adj = [], []
    pending_nodes = 0

    def flush():
        nonlocal pending_ids, pending_adj, pending_nodes
        if not pending_ids:
            return
        with torch.inference_mode():
            batch = model(
                [ids.to(device, non_blocking=True) for ids in pending_ids],
                pending_adj,
            ).cpu()
        embeddings.extend(batch.unbind(0))
        pending_ids, pending_adj, pending_nodes = [], [], 0

    for key in keys:
        index = dataset.mapping[key]
        ids, adj = dataset.process_function(dataset.dataset[index])
        nodes = ids.size(0)
        if pending_ids and pending_nodes + nodes > node_budget:
            flush()
        pending_ids.append(ids)
        pending_adj.append(adj)
        pending_nodes += nodes
        if nodes >= node_budget:
            flush()
    flush()
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
    anchor_embeddings = encode_keys(
        model, dataset, device, [anchor for anchor, _ in retrieval])
    candidate_embeddings = F.normalize(candidate_embeddings, dim=1)
    anchor_embeddings = F.normalize(anchor_embeddings, dim=1)
    iterator = progress(range(len(retrieval))) if progress is not None \
        else range(len(retrieval))
    reciprocal_ranks = []
    rank1 = []
    for index in iterator:
        _, true_positions = retrieval[index]
        similarities = torch.mv(
            candidate_embeddings, anchor_embeddings[index])
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
