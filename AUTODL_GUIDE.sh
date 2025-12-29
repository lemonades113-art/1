#!/bin/bash
# ==============================================================================
# grpo_math_verl - AutoDL 训练完整指南
# ==============================================================================
# 项目：数学推理 GRPO 训练
# 模型：Qwen2.5-Math-7B-Instruct
# 数据：GSM8K (7.4k条) 或 OpenR1-Math (93.7k条)
# 硬件：4卡 GPU (推荐 A100 80G 或 4090 24G x4)
# ==============================================================================

# ==========================
# Step 1: 上传后的初始设置
# ==========================

# 解压项目
cd /root/autodl-tmp
unzip grpo_math_verl.zip
cd grpo_math_verl

# 安装依赖
pip install -r requirements.txt
pip install wandb accelerate deepspeed peft transformers datasets tqdm

# ==========================
# Step 2: 配置 WandB
# ==========================
export WANDB_API_KEY="6cb4e63f9868799160fa077676721e37704709b9"
wandb login --relogin $WANDB_API_KEY

# ==========================
# Step 3: 配置 HuggingFace 镜像（加速下载）
# ==========================
export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM=false

# ==========================
# Step 4: 下载数据集（二选一）
# ==========================

# 方式A: 使用本地 GSM8K 数据（已包含在项目中）
# 无需操作，数据在 data/gsm8k/train_full_7473.json

# 方式B: 下载 OpenR1-Math 数据集（约9万条，推荐！）
python download_openr1_math.py --dataset openr1 --sample_size 30000 --output_dir ./data/openr1_math

# ==========================
# Step 5: 检查 GPU 状态
# ==========================
nvidia-smi
# 确认有4卡可用

# ==========================
# Step 6: 运行演示测试（可选）
# ==========================
python main.py --mode demo

# ==========================
# Step 7: 启动4卡分布式训练
# ==========================

# 使用 GSM8K 数据训练
accelerate launch --config_file accelerate_config.yaml main.py --mode train --sources gsm8k --algorithm grpo

# 或使用 OpenR1 数据训练
accelerate launch --config_file accelerate_config.yaml main.py --mode train --sources openr1 --algorithm grpo

# 可选算法：grpo, gspo, rloo

# ==========================
# Step 8: 监控训练
# ==========================
# WandB 会自动记录：
# - train/loss
# - train/accuracy  
# - train/mean_reward
# 
# 打开 https://wandb.ai/ 查看训练曲线

# ==========================
# Step 9: 评估模型
# ==========================
python main.py --mode eval --model_path ./outputs/grpo_math_verl/final

# ==========================
# Step 10: 下载训练好的模型
# ==========================
# 模型保存在: ./outputs/grpo_math_verl/final/
# 可以打包下载或推送到 HuggingFace Hub
