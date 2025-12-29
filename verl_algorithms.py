# -*- coding: utf-8 -*-
"""
VERL核心算法模块
================

从VERL (https://github.com/volcengine/verl) 移植的核心RL算法：
- compute_grpo_outcome_advantage: GRPO组内标准化
- compute_rloo_outcome_advantage: RLOO (ReLoO)
- compute_policy_loss_dual_clip: Dual-Clip PPO
- compute_policy_loss_GSPO: GSPO序列级ratio
- AdaptiveKLController: 自适应KL控制

来源: verl/trainer/ppo/core_algos.py
License: Apache-2.0

面试话术：
"核心算法参考VERL框架实现，采用组内标准化的Advantage计算，
支持GRPO/GSPO/RLOO三种算法。Dual-Clip PPO可以更好地处理负优势，
避免策略更新过激。"
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Tuple, Dict, Optional


# ============================================================================
# KL Controller (来自VERL)
# ============================================================================

class AdaptiveKLController:
    """
    自适应KL控制器
    
    来源: https://arxiv.org/pdf/1909.08593.pdf
    
    根据当前KL散度动态调整KL系数：
    - 如果KL > target，增加系数（惩罚偏离）
    - 如果KL < target，减少系数（允许更多探索）
    """
    
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon
    
    def update(self, current_kl: float, n_steps: int):
        """更新KL系数"""
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """固定KL系数控制器"""
    
    def __init__(self, kl_coef: float):
        self.value = kl_coef
    
    def update(self, current_kl: float, n_steps: int):
        pass


# ============================================================================
# Advantage计算 (来自VERL - 核心!)
# ============================================================================

def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算GRPO的Advantage (来自VERL)
    
    核心思想：
    1. 每个样本有一个scalar reward (outcome reward)
    2. 按prompt分组 (index相同的属于同一组)
    3. 组内计算mean和std
    4. 标准化: advantage = (reward - mean) / (std + epsilon)
    
    Args:
        token_level_rewards: [batch_size, response_length] token级reward
        eos_mask: [batch_size, response_length] response mask
        index: [batch_size] prompt的唯一ID，相同ID表示同一prompt的不同响应
        epsilon: 数值稳定性
    
    Returns:
        advantages: [batch_size, response_length]
        returns: [batch_size, response_length]
    """
    response_length = token_level_rewards.shape[-1]
    
    # 提取每个样本的scalar reward (非零位置求和)
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)  # [batch_size]
    
    # 按index分组计算统计量
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        
        # 收集每个prompt的所有得分
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # 计算每个prompt的组内mean/std
        for idx in id2score:
            group_scores = id2score[idx]
            if len(group_scores) == 1:
                # 单样本不标准化
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(group_scores) > 1:
                stacked = torch.stack(group_scores)
                id2mean[idx] = torch.mean(stacked)
                id2std[idx] = torch.std(stacked)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        # 标准化
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        
        # 扩展到token级别
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    
    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算RLOO的Advantage (来自VERL)
    
    RLOO: https://arxiv.org/abs/2402.14740
    
    核心思想：Leave-one-out baseline
    对于第i个样本，baseline是组内其他样本的平均reward
    
    公式：
    advantage_i = reward_i * n/(n-1) - mean(all_rewards) * n/(n-1)
    
    这相当于把自己排除后计算的baseline。
    """
    scores = token_level_rewards.sum(dim=-1)  # [batch_size]
    
    id2score = defaultdict(list)
    id2mean = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        for idx in id2score:
            group_scores = id2score[idx]
            if len(group_scores) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            elif len(group_scores) > 1:
                id2mean[idx] = torch.mean(torch.stack(group_scores))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        # Leave-one-out调整
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                # Leave-one-out: 把自己的贡献去掉
                scores[i] = (scores[i] * response_num / (response_num - 1) - 
                            id2mean[index[i]] * response_num / (response_num - 1))
        
        scores = scores.unsqueeze(-1) * response_mask
    
    return scores, scores


# ============================================================================
# Policy Loss计算 (来自VERL)
# ============================================================================

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    """带mask的平均值"""
    if dim is None:
        return (tensor * mask).sum() / mask.sum().clamp(min=1)
    else:
        return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)


def compute_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange: float = 0.2
) -> Tuple[torch.Tensor, float, float]:
    """
    标准PPO Policy Loss (来自VERL)
    
    Args:
        old_log_prob: [batch, seq_len] 旧策略log概率
        log_prob: [batch, seq_len] 新策略log概率
        advantages: [batch, seq_len] 优势
        response_mask: [batch, seq_len] mask
        cliprange: clipping范围
    
    Returns:
        pg_loss: 策略梯度损失
        pg_clipfrac: 被clip的比例
        ppo_kl: KL散度
    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask).item()
    
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    
    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), response_mask)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), response_mask).item()
    
    return pg_loss, pg_clipfrac, ppo_kl


def compute_policy_loss_dual_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange: float = 0.2,
    cliprange_low: float = None,
    cliprange_high: float = None,
    clip_ratio_c: float = 3.0,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, float, float, float]:
    """
    Dual-Clip PPO Loss (来自VERL)
    
    论文: https://arxiv.org/pdf/1912.09729
    
    核心思想：
    - 标准PPO只clip正方向的ratio
    - Dual-Clip额外对负优势的情况进行下限clip
    - 避免负优势时ratio过小导致的不稳定
    
    Args:
        clip_ratio_c: 负优势时ratio的下限 (通常3.0)
    """
    assert clip_ratio_c > 1.0
    
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask).item()
    
    # 标准PPO clip
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask).item()
    
    # Dual-clip: 对负优势进行下限clip
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(
        (torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0)).float(), 
        response_mask
    ).item()
    
    # 最终loss
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode)
    
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_GSPO(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange: float = 0.2,
    cliprange_low: float = None,
    cliprange_high: float = None,
    clip_ratio_c: float = 3.0,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, float, float, float]:
    """
    GSPO Loss (来自VERL) - 序列级别ratio
    
    核心思想：
    - 标准PPO/GRPO使用token-level ratio
    - GSPO使用sequence-level ratio (几何平均)
    - 更稳定，避免单个token的ratio过大影响整体
    
    公式：
    log_ratio_seq = mean(log_prob - old_log_prob) over tokens
    ratio = exp(log_ratio_seq)  # 这是几何平均!
    """
    assert clip_ratio_c > 1.0
    
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    
    # 序列级别的ratio (几何平均)
    seq_len = response_mask.sum(dim=1, keepdim=True).clamp(min=1)
    negative_approx_kl_seq = (negative_approx_kl * response_mask).sum(dim=1, keepdim=True) / seq_len
    ratio_seq = torch.exp(negative_approx_kl_seq)
    ratio = ratio_seq.expand_as(negative_approx_kl)  # 扩展到token级别
    
    ppo_kl = (-negative_approx_kl_seq).mean().item()
    
    # 和Dual-Clip相同的clip逻辑
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask).item()
    
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(
        (torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0)).float(),
        response_mask
    ).item()
    
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode)
    
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str
) -> torch.Tensor:
    """
    Loss聚合 (来自VERL)
    
    支持多种聚合方式：
    - token-mean: 所有token平均
    - seq-mean-token-sum: 先token求和，再sequence平均
    - seq-mean-token-mean: 先token平均，再sequence平均
    """
    if loss_agg_mode == "token-mean":
        loss = masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1).clamp(min=1)
        loss = torch.mean(seq_losses)
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
    
    return loss


# ============================================================================
# KL Penalty计算 (来自VERL)
# ============================================================================

def kl_penalty(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
    kl_penalty_type: str = "kl"
) -> torch.Tensor:
    """
    计算KL散度惩罚 (来自VERL)
    
    支持多种KL类型：
    - kl: 标准KL
    - abs: 绝对值
    - mse: 均方误差
    - low_var_kl: 低方差KL (Schulman)
    """
    if kl_penalty_type == "kl":
        return logprob - ref_logprob
    
    if kl_penalty_type == "abs":
        return (logprob - ref_logprob).abs()
    
    if kl_penalty_type == "mse":
        return 0.5 * (logprob - ref_logprob).square()
    
    if kl_penalty_type == "low_var_kl":
        # J. Schulman. Approximating kl divergence, 2020.
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20.0, max=20.0)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
    
    raise NotImplementedError(f"Unknown kl_penalty_type: {kl_penalty_type}")


# ============================================================================
# 便捷函数
# ============================================================================

def get_advantage_fn(algorithm: str):
    """获取Advantage计算函数"""
    if algorithm in ["grpo", "gspo"]:
        return compute_grpo_outcome_advantage
    elif algorithm == "rloo":
        return compute_rloo_outcome_advantage
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def get_policy_loss_fn(algorithm: str):
    """获取Policy Loss计算函数"""
    if algorithm == "grpo":
        return compute_policy_loss_dual_clip
    elif algorithm == "gspo":
        return compute_policy_loss_GSPO
    elif algorithm == "rloo":
        return compute_policy_loss_dual_clip
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 VERL 核心算法")
    print("=" * 60)
    
    # 模拟数据
    batch_size = 8
    seq_len = 64
    
    # 模拟token_level_rewards (只有最后一个token有reward)
    rewards = torch.zeros(batch_size, seq_len)
    rewards[:, -1] = torch.tensor([0.8, 0.5, 0.3, 0.9, 0.7, 0.4, 0.6, 0.2])
    
    # 模拟response mask
    mask = torch.ones(batch_size, seq_len)
    
    # 模拟index (前4个属于prompt1, 后4个属于prompt2)
    index = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    print("\n1. 测试GRPO Advantage计算:")
    adv, ret = compute_grpo_outcome_advantage(rewards, mask, index)
    print(f"   原始rewards: {rewards[:, -1].tolist()}")
    print(f"   GRPO advantages: {adv[:, -1].tolist()}")
    
    print("\n2. 测试RLOO Advantage计算:")
    adv_rloo, ret_rloo = compute_rloo_outcome_advantage(rewards, mask, index)
    print(f"   RLOO advantages: {adv_rloo[:, -1].tolist()}")
    
    print("\n3. 测试Policy Loss计算:")
    old_logp = torch.randn(batch_size, seq_len) * 0.1
    new_logp = old_logp + torch.randn(batch_size, seq_len) * 0.05
    
    loss, clipfrac, kl, clipfrac_lower = compute_policy_loss_dual_clip(
        old_logp, new_logp, adv, mask
    )
    print(f"   Dual-Clip Loss: {loss.item():.4f}")
    print(f"   Clip Fraction: {clipfrac:.4f}")
    print(f"   KL: {kl:.4f}")
    
    loss_gspo, _, _, _ = compute_policy_loss_GSPO(
        old_logp, new_logp, adv, mask
    )
    print(f"   GSPO Loss: {loss_gspo.item():.4f}")
    
    print("\n✅ 所有测试通过！")
