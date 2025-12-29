# -*- coding: utf-8 -*-
"""
GRPO Math 主程序 (VERL风格重构)
================================

Usage:
    # 单卡演示
    python main.py --mode demo
    
    # 4卡训练
    accelerate launch --config_file accelerate_config.yaml main.py --mode train
    
    # 评估
    python main.py --mode eval --model_path ./outputs/grpo_math_verl/final

面试话术：
"使用accelerate + DeepSpeed ZeRO-2实现4卡分布式训练，
核心算法采用VERL的组内标准化Advantage计算。
支持GRPO/GSPO/RLOO三种算法切换，7B模型4卡可训练。"
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

from config import CONFIG, get_config
from data_module import MathDataset, MathProblem
from reward_function import MathRewardFunction, RewardTracker
from grpo_trainer import VerlGRPOTrainer


def setup_environment():
    """环境准备"""
    dirs = ["./data", "./cache", "./outputs", "./logs"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ 环境准备完成")


def load_data(
    sources: List[str] = None,
    max_train: int = None,
    max_eval: int = None
) -> MathDataset:
    """加载数据"""
    dataset = MathDataset(
        sources=sources or ["gsm8k"],
        max_train=max_train or CONFIG.data.max_train_samples,
        max_eval=max_eval or CONFIG.data.max_eval_samples
    )
    dataset.load()
    return dataset


def run_demo():
    """快速演示"""
    print("\n" + "=" * 60)
    print("🎯 GRPO Math 演示 (VERL风格)")
    print("=" * 60)
    
    # 加载少量数据
    dataset = load_data(sources=["gsm8k"], max_train=50, max_eval=10)
    
    # 测试VERL算法
    print("\n🔍 测试VERL核心算法:")
    from verl_algorithms import compute_grpo_outcome_advantage, compute_policy_loss_dual_clip
    import torch
    import numpy as np
    
    # 模拟数据
    rewards = torch.tensor([[0.8], [0.5], [0.3], [0.9]])  # 4个样本
    mask = torch.ones(4, 1)
    index = np.array([0, 0, 1, 1])  # 2个prompt，每个2个响应
    
    adv, _ = compute_grpo_outcome_advantage(rewards, mask, index)
    print(f"  原始rewards: {rewards[:, 0].tolist()}")
    print(f"  GRPO advantages: {adv[:, 0].tolist()}")
    
    # 测试Reward函数
    print("\n🔍 测试Reward函数:")
    reward_fn = MathRewardFunction()
    test_cases = [
        ("Let me solve step by step. 2+2=4. #### 4", "4"),
        ("#### 4", "4"),
        ("#### 5", "4"),
    ]
    for response, gold in test_cases:
        result = reward_fn.compute(response, gold)
        print(f"  '{response[:30]}...' → reward={result.total:.3f}, correct={result.is_correct}")
    
    print("\n✅ 演示完成！完整训练请使用:")
    print("   accelerate launch --config_file accelerate_config.yaml main.py --mode train")


def run_training(algorithm: str = "grpo", sources: List[str] = None):
    """运行训练"""
    print("\n" + "=" * 60)
    print(f"🚀 {algorithm.upper()} Math 训练 (VERL风格, 分布式)")
    print("=" * 60)
    
    # 加载数据
    dataset = load_data(sources=sources or ["gsm8k"])
    train_data = dataset.get_train_dataset()
    
    # 创建训练器
    trainer = VerlGRPOTrainer(algorithm=algorithm)
    
    # 初始化
    if not trainer.setup():
        print("❌ 初始化失败")
        return
    
    # 训练
    trainer.train(
        train_data=train_data,
        num_epochs=CONFIG.grpo.num_epochs,
        batch_size=CONFIG.grpo.batch_size,
    )
    
    print("\n✅ 训练完成！")
    print(f"   模型保存到: {trainer.output_dir}")


def run_evaluation(model_path: str = None):
    """运行评估"""
    print("\n" + "=" * 60)
    print("📊 GRPO Math 评估")
    print("=" * 60)
    
    dataset = load_data(sources=["gsm8k"], max_eval=100)
    
    if model_path and Path(model_path).exists():
        print(f"   使用模型: {model_path}")
        # TODO: 加载模型进行评估
    else:
        print("   使用Reward函数评估 (无模型)")
    
    # 简单统计
    print(f"\n   评估集大小: {len(dataset.eval_data)}")
    print("   (完整评估需要加载训练后的模型)")


def main():
    parser = argparse.ArgumentParser(description="GRPO Math Training (VERL-style)")
    parser.add_argument(
        "--mode",
        choices=["demo", "eval", "train"],
        default="demo",
        help="运行模式"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="评估时使用的模型路径"
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="gsm8k",
        help="数据源（gsm8k,openr1）"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["grpo", "gspo", "rloo"],
        default="grpo",
        help="训练算法"
    )
    
    args = parser.parse_args()
    
    setup_environment()
    sources = args.sources.split(",")
    
    if args.mode == "demo":
        run_demo()
    elif args.mode == "eval":
        run_evaluation(args.model_path)
    elif args.mode == "train":
        run_training(algorithm=args.algorithm, sources=sources)


if __name__ == "__main__":
    main()
