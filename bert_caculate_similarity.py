"""Evaluation of the fine-tuned CFGFusionModel on the cross-platform
function similarity task (paper task 1): MRR10 and Rank1.

Every eligible x64 test function is ranked against all ARM64 functions from
the same optimization-level test set. All compiler-version variants with the
same symbol and binary target are relevant answers.

Usage:
    uv run python bert_caculate_similarity.py [--checkpoint PATH] [--device cuda] [--batch_size 20]
"""
import argparse
import os
import torch
from tqdm import tqdm

from models.tokenizer import AsmTokenizer
from models.bert import BERTForPretraining
from models.model import CFGFusionModel
from models.dataset import FunctionPairDataset
from models.retrieval import (
    build_retrieval_sets, encode_keys, evaluate_retrieval, load_function_keys,
)
from models.trainer import _resolve_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate CFGFusionModel (MRR10/Rank1)")
    parser.add_argument("--pool", default=os.path.join(
        "outputs", "task1-o2-test-function_pool.csv"),
        help="test pair pool CSV (e.g. task1-o2-test-function_pool.csv)")
    parser.add_argument("--checkpoint", default=os.path.join(
        "outputs", "bert4-finetune-hf", "CFGFusion-best.pth"))
    parser.add_argument("--pretrained", default=os.path.join(
        "outputs", "bert4-pretrain-hf", "bert-best"),
        help="pre-trained BERT checkpoint directory")
    parser.add_argument("--d_model", type=int, default=128,
                        help="BERT hidden size (must match the pre-trained model)")
    parser.add_argument("--mpnn_readout_dim", type=int, default=64)
    parser.add_argument("--cnn_out", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_finetune", action="store_true",
                        help="use the raw pre-trained embeddings without the "
                             "fine-tuned checkpoint (ablation)")
    parser.add_argument("--candidate_pool", default=os.path.join(
        "outputs", "task2-arm64-test-functions.pkl"),
        help="candidate function list (6-tuple keys)")
    parser.add_argument("--anchor_pool", default=os.path.join(
        "outputs", "task2-x64-test-functions.pkl"),
        help="anchor function list (6-tuple keys)")
    args = parser.parse_args()
    device = _resolve_device(args.device)

    tokenizer = AsmTokenizer(vocab_file=os.path.join("outputs", "baseline-vocab.txt"))
    dataset = FunctionPairDataset(
        function_pool_path=args.pool,
        dataset_path=os.path.join("outputs", "baseline-test.jsonl"),
        function_idx_mapping_path=os.path.join(
            "outputs", "test-function-idx-mapping.pkl"),
        tokenizer=tokenizer,
        seq_len=128,
    )

    bert = BERTForPretraining.from_pretrained(args.pretrained, device=device)
    model = CFGFusionModel(bert, d_model=args.d_model,
                           mpnn_readout_dim=args.mpnn_readout_dim,
                           cnn_out=args.cnn_out,
                           hidden_dim=args.hidden_dim).to(device)
    if not args.no_finetune:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                         weights_only=True))
    model.eval()
    if args.no_finetune:
        print("Using raw pre-trained embeddings (no fine-tuning)")
    else:
        print(f"Loaded checkpoint from {args.checkpoint}")

    opt_level = None
    for token in ("task1-o2", "task1-o3"):
        if token in args.pool:
            opt_level = token.replace("task1-", "").upper()
            break
    evaluate_full_pool(
        model, dataset, device, args.anchor_pool, args.candidate_pool, opt_level)


def evaluate_full_pool(model, dataset, device, anchor_pool_path,
                       candidate_pool_path, opt_level=None):
    """Paper-style retrieval: rank each x64 anchor against ALL arm64 test
    functions by cosine similarity of graph embeddings."""
    anchor_keys = load_function_keys(anchor_pool_path)
    candidate_keys = load_function_keys(candidate_pool_path)
    retrieval, cand_keys = build_retrieval_sets(
        anchor_keys, candidate_keys, opt_level)
    print(f"Candidate pool: {len(cand_keys)} ({opt_level or 'all opts'}) ARM64 functions")
    print(f"Eligible anchors: {len(retrieval)}")

    metrics = evaluate_retrieval(
        model, dataset, device, anchor_keys, candidate_keys, opt_level,
        progress=lambda items: tqdm(items, desc="Ranking anchors (full pool)"),
    )
    print(f"\n[full pool] Test anchors: {metrics['anchors']}")
    print(f"[full pool] MRR10: {metrics['mrr10']:.4f}")
    print(f"[full pool] Rank1: {metrics['rank1']:.4f}")



if __name__ == "__main__":
    main()
