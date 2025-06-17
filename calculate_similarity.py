import pandas as pd
import os
import argparse
from models.dataset import FunctionPairDataset
from models.tokenizer import AsmTokenizer
import torch
from torch.utils.data import DataLoader
from models.bert import BERT2
from models.model import SemanticAwareModel, SiameseNetwork


if __name__ == "__main__":
    data_dir = "."
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--device", default="cuda", dest="device")
    parser.add_argument("--batch_size", type=int, default=20, dest="batch_size")
    args = parser.parse_args()
    seq_len = 128
    tokenizer = AsmTokenizer(
        vocab_file=os.path.join(data_dir, "outputs", f"baseline-vocab.txt")
    )
    csv_path="outputs/test-function_pool.csv",
    cache_dir = os.path.join(data_dir, "outputs", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    # Create datasets
    test_dataset = FunctionPairDataset(
        csv_path=csv_path,
        jsonl_path="outputs/baseline-test.jsonl",
        mapping_path="outputs/test-function-idx-mapping.pkl",
        tokenizer=tokenizer,
        seq_len=128,
        max_blocks=50
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
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
    semantic_model = SemanticAwareModel(
        bert_model=bert_model,
        d_model=128,
        hidden_dim=64,
        device=args.device
    )
    model = SiameseNetwork(semantic_model, 64)
    model.load_state_dict(
        torch.load(
            os.path.join(data_dir, "outputs", "semantic_model.pth")
        )
    )
    model = model.to(args.device)
    model.eval()
    correct = 0
    total = 0
    predictions_collect = []
    output_csv_path = os.path.join(data_dir, "outputs", f"baseline-results.csv")
    with torch.no_grad():
        for a_ids, a_adj, t_ids, t_adj, labels in test_data_loader:
            a_ids, a_adj = a_ids.to(args.device), a_adj.to(args.device)
            t_ids, t_adj = t_ids.to(args.device), t_adj.to(args.device)
            labels = labels.to(torch.int32)
            outputs = model(a_ids, a_adj, t_ids, t_adj).cpu()
            predictions = (outputs > 0.5).to(torch.int32)
            predictions_collect.extend(predictions.numpy().tolist())
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    # Save predictions to CSV
    function_pool = pd.read_csv(csv_path)
    function_pool["prediction"] = predictions_collect
    function_pool.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

