#!/usr/bin/env bash
# ============================================================
# 一键全流程训练（Order-Matters 复现）
#
# 自动执行：normalize → split → 预训练(10ep) → Task1 微调(O2/O3)
#           → 全池评估 → Task2 分类(x64/arm64) → 评估 → 汇总
#
# 幂等设计：已完成阶段自动跳过（断点续跑）；预训练重跑后，
# 所有下游阶段（微调/评估/Task2）按时间戳自动重跑。
# 中断后重新执行本脚本即可从断点继续。
#
# 用法：
#   bash train.sh               # 跑全流程（推荐，晚上挂机）
#   bash train.sh --force pretrain   # 强制重跑某阶段
#   nohup bash train.sh > /tmp/train.log 2>&1 &   # 后台挂机
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

# Keep PyTorch's CUDA allocator on fixed segments for reproducible peak-memory
# stress tests and training, as requested. Every pipeline subprocess inherits it.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
# The slow HF tokenizer otherwise writes one informational truncation warning
# per sequence pair (millions of lines) even though truncation is intentional.
export TRANSFORMERS_VERBOSITY="error"

LOG_DIR="outputs/pipeline-logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/pipeline-$STAMP.log"

echo "[train.sh] $(date '+%F %T') 一键训练启动，日志: $LOG_FILE"
echo "[train.sh] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "[train.sh] 当前为完整训练 split；总时长将显著高于旧版 5% 抽样配置，请按日志观察"
echo "[train.sh] 中断后重跑本脚本即断点续跑"

# 清理 datasets 缓存残留锁（进程被强杀后可能遗留，导致 load_dataset 卡死）
find outputs/cache -name "*.lock" -delete 2>/dev/null || true

# pipeline.py 自带幂等跳过与汇总打印；tee 同时落盘
uv run python -u pipeline.py "$@" 2>&1 | tee "$LOG_FILE"
