import pandas as pd
import os
from collections import defaultdict
import gc
import argparse
import pickle
from utils.similarity_metrics import cosine_similarity, L1_Loss
from models.dataset import BERTMLMDataset
from models.tokenizer import AsmTokenizer
import torch
from torch.utils.data import DataLoader
from models.bert import BERT2
from datasets import load_dataset


def get_bert_output(data, model, tokenizer):
    test_dataset = BERTMLMDataset(data['instruction_blocks'], tokenizer, max_len=128, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    model.eval()

    temp_embedding = []
    with torch.no_grad():
        for batch_input in test_dataloader:
            temp = model.encode(batch_input[0].to(model.device))
            temp_embedding.append(temp.cpu())

    embedding = torch.cat(temp_embedding, dim=0)
    return embedding.mean(dim=0).cpu()


if __name__ == "__main__":
    data_dir = "."
    csv_path = os.path.join(data_dir, "outputs", "function_pool.csv")
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--chunksize", default=10000, type=int, dest="chunksize")
    parser.add_argument("--device", default="cuda", dest="device")
    parser.add_argument("--batch_size", type=int, default=10, dest="batch_size")
    args = parser.parse_args()
    seq_len = 128


    tokenizer = AsmTokenizer(
        vocab_file=os.path.join(data_dir, "outputs", f"baseline-vocab.txt")
    )
    cache_dir = os.path.join(data_dir, "outputs", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(data_dir, "outputs", f"baseline-test.jsonl"),
        split="train",
        cache_dir=cache_dir,
        keep_in_memory=False  # Use memory mapping to save memory
    )
    with open(os.path.join(data_dir, "outputs", f"baseline-test-function_name_idx_map.pkl"), "rb") as f:
        function_name_idx_map = pickle.load(f)
    # mlm_test_dataset = TaskDataset(test_dataset, "mlm")
    bert_model = BERT2(
        vocab_size=len(tokenizer.vocab),
        d_model=128,
        n_layers=12,
        heads=8,
        seq_len=seq_len,
        device=args.device
    )
    epoch = 3
    pretrained_model_path = os.path.join(data_dir, "outputs", "epoch", f"model-epoch-{epoch}.pth")
    bert_model.load_state_dict(
        torch.load(pretrained_model_path)
    )
    bert_model.to(args.device)
    bert_model.eval()
    sum = 0
    count = 0
    output_csv_path = os.path.join(data_dir, "outputs", f"baseline-results-cosine-epoch-{epoch}.csv")
    for chunk in pd.read_csv(
        csv_path,
        chunksize=args.chunksize,
        dtype={"anchor_version": str, "target_version": str},
    ):
        function_pools = defaultdict(list)
        results = []
        for _, row in chunk.iterrows():
            anchors = (
                row["anchor_function_name"],
                row["anchor_compiler"],
                row["anchor_version"],
                row["anchor_opt"],
                row["anchor_function_file"],
            )
            targets = (
                row["target_function_name"],
                row["target_compiler"],
                row["target_version"],
                row["target_opt"],
                row["target_function_file"],
                row["label"]
            )
            function_pools[anchors].append(targets)

        exist_embeddings = dict()

        for a_name, a_compiler, a_ver, a_opt, a_bin in function_pools:
            if (a_name, a_compiler, a_ver, a_opt, a_bin) not in exist_embeddings:
                try:
                    idx = function_name_idx_map[
                        (a_name, a_compiler, a_ver, a_opt, a_bin)
                    ]
                    a_assembly = test_dataset[idx]
                except KeyError:
                    print(
                        f"Warning: missing anchor function: {(a_name, a_compiler, a_ver, a_opt, a_bin)}"
                    )
                    continue
                a_embedding = get_bert_output(a_assembly, bert_model, tokenizer)
                exist_embeddings[(a_name, a_compiler, a_ver, a_opt, a_bin)] = (
                    a_embedding
                )
                if count == 0:
                    print((a_name, a_compiler, a_ver, a_opt, a_bin))
                    print(a_embedding)
            else:
                a_embedding = exist_embeddings[
                    (a_name, a_compiler, a_ver, a_opt, a_bin)
                ]
            for t_name, t_compiler, t_ver, t_opt, t_bin, label in function_pools[
                (a_name, a_compiler, a_ver, a_opt, a_bin)
            ]:
                if (t_name, t_compiler, t_ver, t_opt, t_bin) not in exist_embeddings:
                    try:
                        idx = function_name_idx_map[
                        (a_name, a_compiler, a_ver, a_opt, a_bin)
                    ]
                        t_assembly = test_dataset[idx]
                    except KeyError:
                        print(
                            f"Warning: missing target function: {(t_name, t_compiler, t_ver, t_opt, t_bin)}"
                        )
                        continue
                    t_embedding = get_bert_output(t_assembly, bert_model, tokenizer)
                    if count == 0:
                        print((t_name, t_compiler, t_ver, t_opt, t_bin))
                        print(t_embedding)
                        count += 1
                    exist_embeddings[(t_name, t_compiler, t_ver, t_opt, t_bin)] = (
                        t_embedding
                    )
                else:
                    t_embedding = exist_embeddings[
                        (t_name, t_compiler, t_ver, t_opt, t_bin)
                    ]
                res = cosine_similarity(a_embedding, t_embedding).item()
                results.append(
                    {
                        "anchor_function_bin": a_bin,
                        "anchor_function_name": a_name,
                        "anchor_compiler": a_compiler,
                        "anchor_version": a_ver,
                        "anchor_opt": a_opt,
                        "target_function_bin": t_bin,
                        "target_function_name": t_name,
                        "target_compiler": t_compiler,
                        "target_version": t_ver,
                        "target_opt": t_opt,
                        "cosine_similarity": res,
                        "label": label,
                    }
                )
        output_temp = pd.DataFrame(results)
        output_temp.to_csv(
            output_csv_path,
            mode="a",
            header=not os.path.exists(output_csv_path),
            index=False,
        )
        sum += len(results)
        print(f"write {sum} samples in total")
        del function_pools, exist_embeddings, results
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
