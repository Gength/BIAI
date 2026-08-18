# AGENTS.md — BIAI

复现 **"Order Matters: Semantic-Aware Neural Networks for Binary Code Similarity Detection"**（AAAI 2020，Tencent Keen Lab）：对二进制函数 CFG，用 **BERT 4 任务预训练**（MLM/ANP/BIG/GC）提取块语义、**MPNN(GRU)** 提取结构、**11 层 ResNet**（作用于邻接矩阵）提取节点顺序，融合为图嵌入后以 siamese 做跨平台函数相似性检测（Task 1）与优化级别分类（Task 2）。Python 3.12 + PyTorch 2.13 + transformers 5.15，uv 管理环境（RTX 4080 16GB）。

## Pipeline 与论文对照

| 环节 | 论文 | 本实现 |
|---|---|---|
| 数据 | Gemini 数据集（x86-64 & ARM，gcc） | 课程 Dataset-1（同源 7 项目 × 6 架构 × gcc/clang × O0-Os），取 gcc+{x64,arm64}+{O0-O3} 子集，**z3 排除**（12GB 过大） |
| CFG 提取 | 论文未提（Gemini 用 IDA） | angr `CFGFast`，块级容错（坏指令标 `invalid`） |
| 块 token | "tokens as words"（反对手工特征） | **原始 token**：mnemonic/寄存器原文，仅无限值→`<IMM>`/`<TARGET>`；词表仅从 train split 重建（弃用旧版 36-token 手工分类） |
| 语义/结构/顺序 | BERT 4 任务 / MPNN(GRU,T=5,sum readout) / ResNet11+global maxpool | 同；BERT 为 HF `BertModel`（128/12 层/8 heads/FFN 256/seq 128） |
| 融合 | `g_final = MLP([g_ss, g_o])` 64 维 | 同 |
| Task1 | x86-64↔ARM siamese+cosine，MRR10/Rank1 | 同（O2/O3 独立数据集；全 x64/arm64 池；同符号+同 binary target 的所有 gcc 版本均为真目标） |
| Task2 | 图分类 O0-O3，softmax+CE，accuracy | 同（x64/arm64 分平台训练报告） |
| 超参 | Adam lr=1e-4，batch=10，T=5 | 微调同且 weight decay=0；预训练 batch=32（论文未公开预训练细节/轮数） |

## Commands

```bash
uv sync                    # 安装依赖（勿用 pip）
bash train.sh              # 一键全流程（幂等断点续跑；--force <stage> 强制单阶段）
                           # 数据→预训练→Task1(微调+全池评估)→Task2(训练+评估)→汇总
uv run python pipeline.py  # 同上（train.sh 的内核，含日志落盘）
uv run python normalize_instr.py [--force]  # 单步：angr 反汇编 → baseline/*.pkl；全量重提取用 --force
uv run python split_dataset.py              # 单步：→ outputs/ JSONL + 配对 CSV + vocab
uv run python bert4_pretrain.py             # 单步：预训练（resume=True 时从 bert-best 续训）
uv run python bert4_finetune.py --task1_opt o2 [--epochs N]   # Task1 微调
uv run python bert4_task2.py --platform x64 [--eval]          # Task2 分类
uv run python bert_caculate_similarity.py --checkpoint PATH --pool outputs/task1-o2-test-function_pool.csv  # 全池评估（唯一口径）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False uv run python tests/stress_gpu_batch.py --nodes 1000  # Task1 极限显存测试
```

## Architecture

- `normalize_instr.py` — angr CFGFast 提取 CFG + 原始 token 策略；只接收 gcc+x64/arm64 的真实 ELF（排除 angr `*_angr_rtdb` sidecar）；`--force` 失败后按进度标记续跑
- `split_dataset.py` — gcc+{x64,arm64}+{O0-O3} 筛选；从 Dataset-1 顶层目录恢复真实项目；按 `(project,function)` 做 80/10/10 隔离；Task1 按 `(binary target,function)` 配对；Task2 列表；train-only vocab
- `models/tokenizer.py` — `AsmTokenizer(PreTrainedTokenizer)`：HF API + 汇编分词正则
- `models/bert.py` — `build_bert_config`（`_attn_implementation="sdpa"`）+ `BERTForPretraining`（4 任务头，目录式 checkpoint）
- `models/graph_dataset.py` — Task1/Task2 图数据快路径：预分配块 token 矩阵、稀疏邻接跨 PCIe；不改变 token/CFG 语义
- `models/model.py` — `CFGFusionModel`：BERT [CLS] 块嵌入 → MPNN + OrderCNN → MLP；CFG 原生尺寸逐图头、禁止节点 padding/clipping；≤1536 总节点自适应关闭 BERT checkpoint，超出则开启
- `models/trainer.py` — `BERTPretrainTrainer`（4 任务加权、每 5ep 存档）及 Task1 共用 loss/checkpoint 基类、`BucketBatchSampler`、`_resolve_device`
- `models/finetune_trainer.py` — Task1 高吞吐执行：node-budget 内 anchor+target 合并 BERT、常驻 DataLoader workers；loss/有效 batch/逐图图头语义不变
- `models/retrieval.py` — Task1 验证/测试共用的全池真值构造、node-budget 批量编码与 MRR10/Rank1；best checkpoint 按 validation MRR10 选择
- `models/checkpoint_utils.py` — 新训练前自动备份已有 best，并清除旧完成标记
- `bert4_*.py` / `bert_caculate_similarity.py` — 训练与唯一全池评估入口（Config 类承载超参）
- `tests/test_regressions.py` / `tests/stress_gpu_*.py` — CPU 回归测试与 RTX 4080 极限 batch 显存测试
- `RESULTS.md` — 全部实验结果与消融（单独文件维护）

## Conventions

- 论文对齐超参：微调 Adam lr=1e-4、weight decay=0、有效 batch=10（Task1/Task2 均为真实 batch=10）、T=5；预训练 batch=32、固定 lr；优化器统一 `torch.optim.Adam`
- 预训练/Task2 使用完整 train split，不做人为 5%/30k 数据集抽样；每函数每预训练任务最多采样 16 个块/块对（论文要求随机采样若干块）
- **可变尺寸图**：dataset/model 均不对节点 pad/clip；model 对 list 中每个 CFG 原生尺寸独立 forward；同一图的嵌入不得依赖 batch companion；token 序列仍按论文上限 pad/truncate 到 128
- BERT 预训练 checkpoint 是目录（config.json + pytorch_model.bin）；Task1/Task2 checkpoint 是本地 `.pth` 权重文件；新训练自动备份已有 best；各阶段仅在成功结束后写 `train-done*.json`
- 反序列化安全：`torch.load` 一律 `weights_only=True`；pkl 数据（mapping/函数列表/baseline）均为本地自产——**禁止加载外部 pkl/checkpoint**
- 数据产物在 `outputs/`、`baseline/`（gitignore）；合成/冒烟测试产物勿提交
- 旧训练/消融数字因伪节点 padding、同源划分泄漏和检索漏标而**全部作废**；不得再引用“5/8ep 持平”“15ep 过拟合”“10ep 优于 20ep”等旧结论，须等新流水线重验
- 显存/速度：SDPA；CUDA AMP+GradScaler；`BucketBatchSampler`；node budget=900（只拆 forward/backward，保持逻辑 batch loss/step）；Task1/Task2 的 BERT 开 gradient checkpointing，预训练关闭；微调启用 cuDNN benchmark
- CUDA allocator 固定为 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`（`train.sh` 设置并由所有子进程继承）
- RTX 4080 极限实测：Task1 1000+1000 节点、seq128、forward+backward+Adam，checkpoint 开启时 peak reserved 5.035 GiB；预训练 BIG 1024×128、无 checkpoint 时 6.988 GiB；Task1 全局关闭 checkpoint 不安全（512+512 已 7.527 GiB，1000+1000 超过 3.5 分钟未完成）
- 高吞吐路径 RTX 4080 复测：4000-node 极限组（2 对 1000+1000）peak allocated 9.021 GiB / reserved 10.822 GiB；关闭 checkpoint 的阈值最坏形状 1000+536 peak 11.704/12.020 GiB；10×(32+32) 从逐图 14.37 提升至 59.24 pairs/s（4.12×），其中自适应 checkpoint 额外提升 16%
- 预训练任务：MLM 为**单块**输入（论文 "token sequences inside the node"）；ANP/BIG/GC 输入**不做 mask**（论文 ANP 无 mask 描述）

## 已知限制

- 论文未公开预训练/微调轮数；当前流水线显式配置为预训练 10ep、Task1 8ep、Task2 15ep，不能宣称是论文原始设置或已验证最优
- **论文 Table 2/3 对比方法未复现**：WL/Gemini/word2vec/skip-thought/CNN-only/MPNN_ws 等无实现（已清理），报告中标注缺失
- z3 项目排除（12GB，需 >16GB 内存才能补跑）
- 评估候选池比论文大 1.8 倍（8.7k vs 4.9k）；gcc 4.8/5/7/9 四版本全保留、数据规模大于论文 Table 1——横向比较需注意
- 数据为课程 Dataset-1（Gemini 同源 7 项目，取 6 项目 gcc x64/arm64 O0-O3 子集）

## Notes

- 2026-08-17 严格对齐修复完成；旧 `outputs/` 已清空。normalize 已验收 1000/1000 个真实 ELF 对应的 baseline（sidecar 误识别故障已修复）；当前没有有效 checkpoint/指标，下一次 `bash train.sh` 会跳过 normalize 并从 split 继续
- pipeline 以代码/数据/上游 checkpoint mtime + 完成标记判定新鲜度；任一依赖变化会自动使下游失效；Task1/Task2 评估结果统一写入 `outputs/results/task1-{o2,o3}.json` 与 `task2-{x64,arm64}.json`
