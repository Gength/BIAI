import pandas as pd
import os
import argparse
from models.dataset import FunctionPairDataset
from models.tokenizer import AsmTokenizer
import torch
from torch.utils.data import DataLoader
from models.bert import BERT2, BERT4
from models.model import CFGFusionModel
from tqdm import tqdm
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--device", default="cuda", dest="device")
    parser.add_argument("--batch_size", type=int, default=150, dest="batch_size")
    args = parser.parse_args()
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
        max_blocks=50
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        prefetch_factor=4,  # Prefetch data for faster loading
        pin_memory=True
    )
    bert_model = BERT2(
        vocab_size=len(tokenizer.vocab),
        d_model=128,
        n_layers=12,
        heads=8,
        seq_len=seq_len,
        device=args.device
    )
    model = CFGFusionModel(
        bert_model=bert_model,
        d_model=128,
        hidden_dim=64,
        device=args.device
    )
    model.load_state_dict(
        torch.load(
            os.path.join("outputs", "bert2-finetune", "CFGFusion-best.pth")
        )
    )
    model = model.to(args.device)
    model.eval()
    correct = 0
    total = 0
    predictions_collect = []
    output_csv_path = os.path.join("outputs", f"baseline-results.csv")
    progress_bar = tqdm(total=len(test_data_loader), desc="Testing", unit="batch")
    with torch.no_grad():
        for a_ids, a_adj, t_ids, t_adj, labels in test_data_loader:
            a_ids, a_adj = a_ids.to(args.device), a_adj.to(args.device)
            t_ids, t_adj = t_ids.to(args.device), t_adj.to(args.device)
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
    # Save predictions to CSV
    function_pool = pd.read_csv(function_pool_path)
    function_pool['label'] = function_pool['label'].astype(int)
    function_pool[function_pool['label'] == 0] = -1 
    function_pool["prediction"] = predictions_collect
    function_pool.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

