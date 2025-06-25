import pandas as pd
import os
from models.dataset import SparseFunctionPairDataset as FunctionPairDataset
from models.tokenizer import AsmTokenizer
import torch
from torch.utils.data import DataLoader
from models.bert import BERT2, BERT4
from models.sparse_model import CFGFusionModel
from tqdm import tqdm
import torch.nn.functional as F
from models.collatefn import sparse_pair_collate_fn

if __name__ == "__main__":
    device = torch.device('cuda')
    batch_size = 2
    seq_len = 128
    tokenizer = AsmTokenizer(
        vocab_file=os.path.join("outputs", f"baseline-vocab.txt")
    )
    function_pool_path = os.path.join("outputs", "test-function_pool.csv")
    cache_dir = os.path.join("outputs", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    # Create datasets
    test_dataset = FunctionPairDataset(
        function_pool_path=function_pool_path,
        dataset_path=os.path.join("outputs", "baseline-test.jsonl"),
        function_idx_mapping_path=os.path.join("outputs", "test-function-idx-mapping.pkl"),
        tokenizer=tokenizer,
        seq_len=128,
        max_nodes=500
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        prefetch_factor=4,  # Prefetch data for faster loading
        pin_memory=True,
        collate_fn=sparse_pair_collate_fn,
    )
    bert_model = BERT2(
        vocab_size=len(tokenizer.vocab),
        d_model=128,
        n_layers=12,
        heads=8,
        seq_len=seq_len,
        device=device
    )
    model = CFGFusionModel(
        bert_model=bert_model,
        d_model=128,
        hidden_dim=64,
        device=device
    )
    model.load_state_dict(
        torch.load(
            os.path.join("outputs", "bert2-finetune", "CFGFusion-best.pth")
        )
    )
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    predictions_collect = []
    output_csv_path = os.path.join("outputs", f"baseline-results.csv")
    progress_bar = tqdm(total=len(test_data_loader), desc="Testing", unit="batch")
    with torch.no_grad():
        for a_ids, a_adj, t_ids, t_adj, labels in test_data_loader:
            if len(a_ids) == 0 or len(t_ids) == 0:
                continue
            if isinstance(a_ids, torch.Tensor):
                a_ids, a_adj = a_ids.to(device), a_adj.to(device)
                t_ids, t_adj = t_ids.to(device), t_adj.to(device)
            labels = labels.to(torch.int)
            a_embeddings = model(a_ids, a_adj)
            t_embeddings = model(t_ids, t_adj)
            cosine_sim = F.cosine_similarity(a_embeddings, t_embeddings, dim=1)
            # add sigmoid activation for binary classification
            predictions = (cosine_sim > 0).to(torch.int).cpu()
            predictions[predictions == 0] = -1
            predictions_collect.extend(predictions.cpu().int().numpy().tolist())
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            current_accuracy = correct / total
            progress_bar.set_postfix({"accuracy": f"{current_accuracy:.4f}"})
            progress_bar.update(1)
    progress_bar.close()
    accuracy = correct / total
    print(f"Final Accuracy: {accuracy:.4f}")

