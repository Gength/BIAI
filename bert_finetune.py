import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from scipy.sparse import coo_matrix
from models.bert import SemanticAwareModel, BERT2
from tqdm import tqdm
from models.tokenizer import AsmTokenizer
from models.dataset import BERTMLMDataset
# 配置参数
class Config:
    batch_size = 5
    max_blocks = 50  # 最大基本块数
    seq_len = 128    # 序列最大长度
    hidden_dim = 64  # 图嵌入维度
    lr = 1e-4
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_checkpoint = vocab_file=os.path.join(".", "outputs", "epoch", "best-model.pth")  # 预训练BERT路径
    save_path = os.path.join(".", "outputs", "semantic_model.pth")  # 模型保存路径

config = Config()

# 自定义数据集类
class FunctionPairDataset(Dataset):
    def __init__(self, csv_path, jsonl_path, mapping_path):
        self.df = load_dataset(
            "csv",
            data_files=csv_path,
            split="train",
            cache_dir=os.path.join(".", "outputs", "cache"),
            keep_in_memory=False
        ).to_pandas(batch_size=config.batch_size)
        self.dataset = load_dataset(
            "json", 
            data_files=jsonl_path,
            split="train",
            cache_dir= os.path.join(".", "outputs", "cache"),
            keep_in_memory=False
        )
        with open(mapping_path, "rb") as f:
            self.mapping = pickle.load(f)
        self.tokenizer = AsmTokenizer(vocab_file=os.path.join(".", "outputs", f"baseline-vocab.txt"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        a_key = (row["anchor_function_name"], row["anchor_compiler"], 
                str(row["anchor_version"]), row["anchor_opt"], row["anchor_function_file"])
        t_key = (row["target_function_name"], row["target_compiler"], 
                str(row["target_version"]), row["target_opt"], row["target_function_file"])
        
        a_idx = self.mapping[a_key]
        t_idx = self.mapping[t_key]
        
        a_data = self.dataset[a_idx]
        t_data = self.dataset[t_idx]
        
        # 处理第一个函数
        a_input_ids, a_adj = self.process_function(a_data)
        # 处理第二个函数
        t_input_ids, t_adj = self.process_function(t_data)
        
        label = torch.tensor(row["label"], dtype=torch.float32)
        return a_input_ids, a_adj, t_input_ids, t_adj, label

    def process_function(self, func_data):
        # 1. 处理指令块
        blocks = func_data["instruction_blocks"]
        tokenized_blocks = []
        for block in blocks:
            tokens = self.tokenizer.encode(block)
            tokens = tokens[:config.seq_len - 2]  # 截断到最大长度
            pad_len = config.seq_len - len(tokens) - 2  # 减去<CLS>和<SEP>
            tokens = [self.tokenizer.vocab['<CLS>']] + tokens + [self.tokenizer.vocab['<SEP>']]
            tokens += [self.tokenizer.vocab['<PAD>']] * pad_len  # 填充到最大长度
            tokenized_blocks.append(torch.tensor(tokens))
        if len(tokenized_blocks) >= config.max_blocks:
            tokenized_blocks = tokenized_blocks[:config.max_blocks]
        else:
            for _ in range(config.max_blocks - len(tokenized_blocks)):
                tokenized_blocks.append(torch.tensor([self.tokenizer.vocab['<PAD>']] * config.seq_len))
        input_ids = torch.stack(tokenized_blocks, dim=0)
        
        # 2. 处理邻接矩阵
        adj_data = func_data["adjacency_matrix"]
        adj = coo_matrix(
            (adj_data["data"], (adj_data["row"], adj_data["col"])),
            shape=(adj_data["shape"])
        ).toarray()

        
        # 调整邻接矩阵大小
        if adj.shape[0] > config.max_blocks:
            adj = adj[:config.max_blocks, :config.max_blocks]
        elif adj.shape[0] < config.max_blocks:
            padded_adj = np.zeros((config.max_blocks, config.max_blocks))
            padded_adj[:adj.shape[0], :adj.shape[1]] = adj
            adj = padded_adj
        
        return input_ids, torch.tensor(adj, dtype=torch.float32)

# 孪生网络模型
class SiameseNetwork(nn.Module):
    def __init__(self, semantic_model):
        super().__init__()
        self.semantic_model = semantic_model
        self.classifier = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, a_ids, a_adj, t_ids, t_adj):
        # 获取图嵌入
        a_embed = self.semantic_model(a_ids, a_adj)
        t_embed = self.semantic_model(t_ids, t_adj)
        
        # 合并嵌入并分类
        combined = torch.cat([a_embed, t_embed], dim=1)
        return self.classifier(combined).squeeze()

# 初始化模型
def init_model(vocab_size):
    # 加载预训练BERT
    bert_model = BERT2(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=12,
        heads=8,
        seq_len=config.seq_len,
        device=config.device
    )
    bert_model.load_state_dict(torch.load(config.bert_checkpoint))
    
    # 创建语义感知模型
    semantic_model = SemanticAwareModel(
        bert_model=bert_model,
        d_model=128,
        hidden_dim=config.hidden_dim,
        device=config.device
    ).to(config.device)
    
    return SiameseNetwork(semantic_model).to(config.device)

# 训练函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc="Training")
    
    for a_ids, a_adj, t_ids, t_adj, labels in progress:
        a_ids, a_adj = a_ids.to(config.device), a_adj.to(config.device)
        t_ids, t_adj = t_ids.to(config.device), t_adj.to(config.device)
        labels = labels.to(config.device)
        
        optimizer.zero_grad()
        outputs = model(a_ids, a_adj, t_ids, t_adj)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    
    return total_loss / len(dataloader)

# 验证函数
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for a_ids, a_adj, t_ids, t_adj, labels in dataloader:
            a_ids, a_adj = a_ids.to(config.device), a_adj.to(config.device)
            t_ids, t_adj = t_ids.to(config.device), t_adj.to(config.device)
            labels = labels.to(config.device)
            
            outputs = model(a_ids, a_adj, t_ids, t_adj)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 计算准确率
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

if __name__ == "__main__":
    # 初始化tokenizer获取词汇表大小
    tokenizer = AsmTokenizer(vocab_file="outputs/baseline-vocab.txt")
    vocab_size = len(tokenizer.vocab)
    
    # 创建数据集
    train_dataset = FunctionPairDataset(
        csv_path="outputs/train-function_pool.csv",
        jsonl_path="outputs/baseline-train.jsonl",
        mapping_path="outputs/train-function-idx-mapping.pkl"
    )
    
    val_dataset = FunctionPairDataset(
        csv_path="outputs/val-function_pool.csv",
        jsonl_path="outputs/baseline-val.jsonl",
        mapping_path="outputs/val-function-idx-mapping.pkl"
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 初始化模型
    model = init_model(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCELoss()  # 二分类交叉熵
    
    best_accuracy = 0
    print(f"Starting training on {config.device}...")
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # 训练
        train_loss = train(model, train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f}")
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), config.save_path)
            print(f"Saved new best model with accuracy {val_acc:.4f}")
    
    print("Training completed!")