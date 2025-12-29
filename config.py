# -*- coding: utf-8 -*-
"""
GRPO Math 配置 (VERL风格重构)
==============================

基于VERL/AReaL架构重构，支持：
- 4卡分布式训练 (accelerate + DeepSpeed ZeRO-2)
- 7B模型 (Qwen2.5-Math-7B-Instruct)
- 完整GRPO/GSPO/RLOO算法

面试话术：
"参考VERL框架重构了训练pipeline，使用DeepSpeed ZeRO-2实现4卡并行，
核心算法采用VERL的组内标准化Advantage计算，支持GRPO/GSPO/RLOO三种算法切换。"
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """模型配置 - 支持7B大模型"""
    # 默认7B模型（4卡可以跑）
    model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    # 备选1.5B模型（资源不足时）
    # model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    
    # 不使用量化（与DeepSpeed兼容）
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    
    # 计算精度
    torch_dtype: str = "bfloat16"  # bf16 for 7B model
    
    # 设备映射（分布式时由accelerate管理）
    device_map: str = None  # None for distributed training


@dataclass
class LoRAConfig:
    """LoRA配置 - 高效微调"""
    use_lora: bool = True
    r: int = 64  # 7B模型用更大的r
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    """数据配置"""
    # 数据源
    train_dataset: str = "gsm8k"  # gsm8k, math, openr1
    eval_dataset: str = "gsm8k"
    
    # OpenR1-Math配置
    openr1_path: str = "open-r1/OpenR1-Math-220k"
    openr1_split: str = "default"  # default split (93.7k条)
    openr1_sample_size: int = 93722  # OpenR1 default split大小
    
    # 数据量
    max_train_samples: int = None  # None = 全量
    max_eval_samples: int = 500
    
    # 序列长度
    max_length: int = 2048  # 7B模型用更长的序列
    max_prompt_length: int = 512
    max_response_length: int = 1536
    
    # 数据路径
    cache_dir: str = "./cache"
    data_dir: str = "./data"


@dataclass
class RewardConfig:
    """Reward函数配置 (参考原项目)"""
    # 正确性权重（最重要）
    correctness_weight: float = 1.0
    
    # 格式奖励
    format_weight: float = 0.2
    cot_bonus: float = 0.1
    
    # 长度控制
    length_weight: float = 0.1
    min_length: int = 50
    max_length: int = 800  # 7B模型允许更长
    optimal_length: int = 300
    
    # 防作弊
    anti_cheat_weight: float = 0.5
    detect_answer_leak: bool = True
    detect_skip_steps: bool = True


@dataclass
class GRPOConfig:
    """GRPO训练配置 (VERL风格)"""
    # 算法选择
    algorithm: str = "grpo"  # grpo, gspo, rloo
    
    # 基础训练参数
    num_epochs: int = 3
    batch_size: int = 4  # per GPU
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5  # 7B模型用较小学习率
    
    # GRPO/GSPO参数 (来自VERL)
    num_generations: int = 4  # 每个prompt生成的候选数 (group_size)
    temperature: float = 1.0  # 采样温度
    
    # KL控制 (来自VERL)
    kl_type: str = "adaptive"  # fixed, adaptive
    kl_coef: float = 0.0  # GRPO通常不用KL
    kl_target: float = 6.0  # for adaptive
    kl_horizon: int = 10000  # for adaptive
    
    # PPO参数 (来自VERL)
    eps_clip: float = 0.4  # clipping range
    eps_clip_low: float = 0.2  # dual-clip lower
    eps_clip_high: float = 0.4  # dual-clip upper
    clip_ratio_c: float = 3.0  # dual-clip c
    
    # Advantage标准化 (来自VERL)
    adv_norm_level: str = "group"  # group, batch
    reward_norm_level: str = "group"  # group, batch
    reward_scaling: float = 10.0  # reward缩放
    reward_bias: float = -0.5  # reward偏移
    
    # Loss聚合 (来自VERL)
    loss_agg_mode: str = "token-mean"  # token-mean, seq-mean-token-sum
    
    # 优化器
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    
    # 显存优化
    gradient_checkpointing: bool = True
    
    # 日志
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    
    # 输出
    output_dir: str = "./outputs/grpo_math_verl"
    
    # WandB配置
    use_wandb: bool = True
    wandb_project: str = "grpo-math-verl"
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_tags: List[str] = field(default_factory=lambda: ["grpo", "verl", "math", "7b"])


@dataclass
class DistributedConfig:
    """分布式训练配置"""
    # GPU数量
    num_gpus: int = 4
    
    # DeepSpeed配置
    deepspeed_stage: int = 2  # ZeRO Stage
    offload_optimizer: bool = False
    offload_param: bool = False
    
    # 混合精度
    mixed_precision: str = "bf16"  # bf16, fp16, no
    
    # 通信
    gradient_as_bucket_view: bool = True
    
    # 推理引擎 (可选vLLM加速)
    use_vllm: bool = False  # 是否用vLLM做生成
    vllm_gpu_memory_utilization: float = 0.8


@dataclass
class EvalConfig:
    """评估配置"""
    pass_at_k: List[int] = field(default_factory=lambda: [1, 5])
    num_samples: int = 5
    eval_temperature: float = 0.0
    sample_temperature: float = 0.7
    eval_size: int = 200


@dataclass
class GRPOMathVerlConfig:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    seed: int = 42
    
    # 项目信息
    project_name: str = "grpo_math_verl"
    description: str = "GRPO Math training with VERL-style algorithms and 4-GPU distributed training"


def get_config() -> GRPOMathVerlConfig:
    """获取配置实例"""
    return GRPOMathVerlConfig()


# 全局配置
CONFIG = get_config()
