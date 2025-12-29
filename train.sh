#!/bin/bash
# ============================================================
# GRPO Math 4卡训练启动脚本 (VERL风格)
# ============================================================
#
# 使用方法:
#   bash train.sh
#   bash train.sh gspo     # 使用GSPO算法
#   bash train.sh rloo     # 使用RLOO算法
#
# 环境要求:
#   - 4张GPU (RTX 5090 32GB 或更大)
#   - Python环境已安装: torch, transformers, accelerate, peft, deepspeed
#
# ============================================================

set -e

# 默认算法
ALGORITHM=${1:-grpo}

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="grpo-math-verl"
export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM="true"

# 项目目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  GRPO Math Training (VERL-style)"
echo "============================================================"
echo "  Algorithm: $ALGORITHM"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "  Config: accelerate_config.yaml"
echo ""

# 检查依赖
python -c "import torch; import transformers; import accelerate; import peft; print('✅ 依赖检查通过')" || {
    echo "❌ 缺少依赖，请安装: pip install -r requirements.txt"
    exit 1
}

# 运行训练
echo "🚀 启动4卡分布式训练..."
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes=4 \
    main.py \
    --mode train \
    --algorithm $ALGORITHM \
    --sources gsm8k

echo ""
echo "============================================================"
echo "  训练完成!"
echo "  模型保存位置: ./outputs/grpo_math_verl"
echo "============================================================"
