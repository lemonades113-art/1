# -*- coding: utf-8 -*-
"""
GRPO 训练器 (VERL风格重构)
===========================

核心特性：
1. 使用VERL的核心算法（GRPO/GSPO/RLOO）
2. 使用accelerate + DeepSpeed ZeRO-2做分布式
3. 支持4卡并行训练7B模型
4. 支持WandB日志

面试话术：
"训练器参考VERL框架重构，核心算法采用组内标准化的Advantage计算，
支持GRPO/GSPO/RLOO三种算法切换。分布式采用DeepSpeed ZeRO-2，
4卡可以训练7B模型。"
"""

import os
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from datetime import datetime

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_scheduler,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from config import CONFIG
from data_module import MathDataset, MathProblem
from reward_function import MathRewardFunction, RewardTracker
from verl_algorithms import (
    compute_grpo_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_policy_loss_dual_clip,
    compute_policy_loss_GSPO,
    AdaptiveKLController,
    FixedKLController,
    get_advantage_fn,
    get_policy_loss_fn,
)


@dataclass
class GenerationResult:
    """生成结果"""
    prompt: str
    responses: List[str]
    rewards: List[float]
    advantages: List[float]
    gold_answer: str
    prompt_id: str


class GRPODataset(Dataset):
    """GRPO训练数据集"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "answer": item["answer"],
            "problem_id": item.get("problem_id", f"p{idx}"),
        }


class VerlGRPOTrainer:
    """
    VERL风格GRPO训练器
    
    支持：
    - GRPO: 组内标准化advantage
    - GSPO: 序列级ratio
    - RLOO: Leave-one-out baseline
    - 4卡分布式 (accelerate + DeepSpeed)
    """
    
    def __init__(
        self,
        model_name: str = None,
        reward_fn: MathRewardFunction = None,
        output_dir: str = None,
        algorithm: str = None,
    ):
        self.model_name = model_name or CONFIG.model.model_name
        self.reward_fn = reward_fn or MathRewardFunction()
        self.output_dir = Path(output_dir or CONFIG.grpo.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.algorithm = (algorithm or CONFIG.grpo.algorithm).lower()
        
        # 模型组件
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        # Accelerator (分布式)
        self.accelerator = None
        
        # KL控制器 (VERL风格)
        if CONFIG.grpo.kl_type == "adaptive":
            self.kl_ctrl = AdaptiveKLController(
                init_kl_coef=CONFIG.grpo.kl_coef,
                target_kl=CONFIG.grpo.kl_target,
                horizon=CONFIG.grpo.kl_horizon
            )
        else:
            self.kl_ctrl = FixedKLController(CONFIG.grpo.kl_coef)
        
        # 追踪器
        self.reward_tracker = RewardTracker()
        self.training_log = []
        self.wandb_initialized = False
        
        # 算法函数
        self.advantage_fn = get_advantage_fn(self.algorithm)
        self.policy_loss_fn = get_policy_loss_fn(self.algorithm)
    
    def setup(self) -> bool:
        """初始化模型和分布式"""
        if not all([HAS_TORCH, HAS_TRANSFORMERS, HAS_ACCELERATE]):
            print("❌ 缺少必要依赖: torch, transformers, accelerate")
            return False
        
        print(f"📦 初始化 VERL-style GRPO Trainer...")
        print(f"   算法: {self.algorithm.upper()}")
        print(f"   模型: {self.model_name}")
        
        # 初始化Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=CONFIG.grpo.gradient_accumulation_steps,
            mixed_precision=CONFIG.distributed.mixed_precision,
            log_with="wandb" if CONFIG.grpo.use_wandb and HAS_WANDB else None,
        )
        
        set_seed(CONFIG.seed)
        
        # 只在主进程打印
        is_main = self.accelerator.is_main_process
        
        if is_main:
            print(f"   GPU数量: {self.accelerator.num_processes}")
            print(f"   混合精度: {CONFIG.distributed.mixed_precision}")
        
        # 加载Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型 (不使用量化，与分布式兼容)
        torch_dtype = getattr(torch, CONFIG.model.torch_dtype) if hasattr(torch, CONFIG.model.torch_dtype) else torch.bfloat16
        
        if is_main:
            print(f"   加载模型中...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            # 不设置device_map，让accelerate管理
        )
        
        # LoRA
        if CONFIG.lora.use_lora and HAS_PEFT:
            if is_main:
                print(f"   应用LoRA (r={CONFIG.lora.r})...")
            
            lora_config = LoraConfig(
                r=CONFIG.lora.r,
                lora_alpha=CONFIG.lora.lora_alpha,
                target_modules=CONFIG.lora.target_modules,
                lora_dropout=CONFIG.lora.lora_dropout,
                bias=CONFIG.lora.bias,
                task_type=CONFIG.lora.task_type,
            )
            self.model = get_peft_model(self.model, lora_config)
            
            if is_main:
                self.model.print_trainable_parameters()
        
        # 梯度检查点
        if CONFIG.grpo.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=CONFIG.grpo.learning_rate,
            weight_decay=CONFIG.grpo.weight_decay
        )
        
        # 用accelerator包装
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        
        if is_main:
            print("✅ 模型初始化完成")
        
        return True
    
    def _setup_wandb(self):
        """初始化WandB"""
        if not CONFIG.grpo.use_wandb or not HAS_WANDB:
            return False
        
        if not self.accelerator.is_main_process:
            return False
        
        try:
            run_name = CONFIG.grpo.wandb_run_name or f"{self.algorithm}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb.init(
                project=CONFIG.grpo.wandb_project,
                entity=CONFIG.grpo.wandb_entity,
                name=run_name,
                tags=CONFIG.grpo.wandb_tags,
                config={
                    "model_name": self.model_name,
                    "algorithm": self.algorithm,
                    "num_gpus": self.accelerator.num_processes,
                    "batch_size": CONFIG.grpo.batch_size,
                    "learning_rate": CONFIG.grpo.learning_rate,
                    "num_generations": CONFIG.grpo.num_generations,
                    "eps_clip": CONFIG.grpo.eps_clip,
                    "use_lora": CONFIG.lora.use_lora,
                    "lora_r": CONFIG.lora.r,
                },
                reinit=True,
            )
            
            self.wandb_initialized = True
            print(f"✅ WandB 初始化: {CONFIG.grpo.wandb_project}/{run_name}")
            return True
        except Exception as e:
            print(f"⚠️ WandB 初始化失败: {e}")
            return False
    
    def _log_wandb(self, metrics: Dict, step: int = None):
        if self.wandb_initialized and self.accelerator.is_main_process:
            wandb.log(metrics, step=step)
    
    def _finish_wandb(self):
        if self.wandb_initialized and self.accelerator.is_main_process:
            wandb.finish()
    
    def generate_responses(
        self,
        prompts: List[str],
        num_generations: int = None,
        temperature: float = None,
    ) -> List[List[str]]:
        """生成响应"""
        num_generations = num_generations or CONFIG.grpo.num_generations
        temperature = temperature or CONFIG.grpo.temperature
        
        all_responses = []
        
        # 获取unwrapped模型用于生成
        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=CONFIG.data.max_prompt_length
            ).to(self.accelerator.device)
            
            responses = []
            for _ in range(num_generations):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=CONFIG.data.max_response_length,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                responses.append(response)
            
            all_responses.append(responses)
        
        model.train()
        return all_responses
    
    def compute_rewards_and_advantages(
        self,
        prompts: List[str],
        responses_batch: List[List[str]],
        gold_answers: List[str],
        prompt_ids: List[str]
    ) -> List[GenerationResult]:
        """计算rewards和advantages (使用VERL算法)"""
        results = []
        
        # 收集所有reward用于VERL风格的组内标准化
        all_rewards = []
        all_indices = []
        
        for i, (prompt, responses, gold, pid) in enumerate(zip(prompts, responses_batch, gold_answers, prompt_ids)):
            for j, response in enumerate(responses):
                result = self.reward_fn.compute(response, gold)
                
                # VERL风格的reward处理
                reward = result.total * CONFIG.grpo.reward_scaling + CONFIG.grpo.reward_bias
                all_rewards.append(reward)
                all_indices.append(i)  # prompt index
                
                self.reward_tracker.add(result)
        
        # 使用VERL的advantage计算
        rewards_tensor = torch.zeros(len(all_rewards), 1)
        rewards_tensor[:, 0] = torch.tensor(all_rewards)
        mask_tensor = torch.ones(len(all_rewards), 1)
        indices_array = np.array(all_indices)
        
        advantages_tensor, _ = self.advantage_fn(
            rewards_tensor, mask_tensor, indices_array
        )
        advantages = advantages_tensor[:, 0].tolist()
        
        # 组装结果
        idx = 0
        for i, (prompt, responses, gold, pid) in enumerate(zip(prompts, responses_batch, gold_answers, prompt_ids)):
            n = len(responses)
            results.append(GenerationResult(
                prompt=prompt,
                responses=responses,
                rewards=all_rewards[idx:idx+n],
                advantages=advantages[idx:idx+n],
                gold_answer=gold,
                prompt_id=pid,
            ))
            idx += n
        
        return results
    
    def compute_policy_loss(
        self,
        results: List[GenerationResult],
    ) -> torch.Tensor:
        """计算策略损失 (使用VERL算法)"""
        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        num_samples = 0
        
        model = self.accelerator.unwrap_model(self.model)
        
        for result in results:
            for response, advantage in zip(result.responses, result.advantages):
                if advantage == 0:
                    continue
                
                # 编码
                full_text = result.prompt + response
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=CONFIG.data.max_length
                ).to(self.accelerator.device)
                
                prompt_inputs = self.tokenizer(result.prompt, return_tensors="pt", truncation=True)
                prompt_len = prompt_inputs["input_ids"].shape[1]
                
                # 前向传播
                with torch.no_grad():
                    old_outputs = model(**inputs)
                    old_logits = old_outputs.logits
                
                outputs = model(**inputs)
                logits = outputs.logits
                
                # 计算log概率
                log_probs = F.log_softmax(logits, dim=-1)
                old_log_probs = F.log_softmax(old_logits, dim=-1)
                
                input_ids = inputs["input_ids"]
                shift_log_probs = log_probs[:, :-1, :]
                shift_old_log_probs = old_log_probs[:, :-1, :]
                shift_labels = input_ids[:, 1:]
                
                response_start = max(0, prompt_len - 1)
                shift_log_probs = shift_log_probs[:, response_start:, :]
                shift_old_log_probs = shift_old_log_probs[:, response_start:, :]
                shift_labels = shift_labels[:, response_start:]
                
                token_log_probs = torch.gather(
                    shift_log_probs, 2, shift_labels.unsqueeze(-1)
                ).squeeze(-1)
                token_old_log_probs = torch.gather(
                    shift_old_log_probs, 2, shift_labels.unsqueeze(-1)
                ).squeeze(-1)
                
                response_mask = torch.ones_like(token_log_probs)
                advantages_tensor = torch.full_like(token_log_probs, advantage)
                
                # 使用VERL的loss函数
                loss, clipfrac, kl, clipfrac_lower = self.policy_loss_fn(
                    token_old_log_probs,
                    token_log_probs,
                    advantages_tensor,
                    response_mask,
                    cliprange=CONFIG.grpo.eps_clip,
                    cliprange_low=CONFIG.grpo.eps_clip_low,
                    cliprange_high=CONFIG.grpo.eps_clip_high,
                    clip_ratio_c=CONFIG.grpo.clip_ratio_c,
                    loss_agg_mode=CONFIG.grpo.loss_agg_mode,
                )
                
                total_loss = total_loss + loss
                num_samples += 1
        
        if num_samples > 0:
            return total_loss / num_samples
        return total_loss
    
    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        """单步训练"""
        prompts = [item["prompt"] for item in batch]
        gold_answers = [item["answer"] for item in batch]
        prompt_ids = [item.get("problem_id", f"p{i}") for i, item in enumerate(batch)]
        
        # 生成
        responses_batch = self.generate_responses(prompts)
        
        # 计算rewards和advantages
        results = self.compute_rewards_and_advantages(
            prompts, responses_batch, gold_answers, prompt_ids
        )
        
        # 计算损失
        loss = self.compute_policy_loss(results)
        
        # 反向传播
        self.accelerator.backward(loss)
        
        if CONFIG.grpo.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), CONFIG.grpo.max_grad_norm)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        if self.scheduler:
            self.scheduler.step()
        
        stats = self.reward_tracker.get_stats()
        
        return {
            "loss": loss.item() if hasattr(loss, 'item') else 0.0,
            "accuracy": stats.get("accuracy", 0),
            "mean_reward": stats.get("mean_reward", 0),
        }
    
    def train(
        self,
        train_data: List[Dict],
        num_epochs: int = None,
        batch_size: int = None,
        eval_fn: Callable = None
    ):
        """完整训练循环"""
        num_epochs = num_epochs or CONFIG.grpo.num_epochs
        batch_size = batch_size or CONFIG.grpo.batch_size
        
        is_main = self.accelerator.is_main_process
        
        if is_main:
            print("\n" + "=" * 60)
            print(f"🚀 开始 {self.algorithm.upper()} 训练 (VERL风格)")
            print("=" * 60)
            print(f"   训练样本: {len(train_data)}")
            print(f"   Epochs: {num_epochs}")
            print(f"   Batch Size: {batch_size} x {self.accelerator.num_processes} GPUs")
        
        self._setup_wandb()
        
        # 学习率调度器
        num_training_steps = (len(train_data) // batch_size) * num_epochs
        self.scheduler = get_scheduler(
            CONFIG.grpo.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=int(num_training_steps * CONFIG.grpo.warmup_ratio),
            num_training_steps=num_training_steps
        )
        
        global_step = 0
        
        for epoch in range(num_epochs):
            if is_main:
                print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
            
            import random
            random.shuffle(train_data)
            
            epoch_losses = []
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                
                with self.accelerator.accumulate(self.model):
                    step_stats = self.train_step(batch)
                
                epoch_losses.append(step_stats["loss"])
                global_step += 1
                
                if global_step % CONFIG.grpo.logging_steps == 0 and is_main:
                    avg_loss = np.mean(epoch_losses[-CONFIG.grpo.logging_steps:])
                    print(f"   Step {global_step}: loss={avg_loss:.4f}, "
                          f"acc={step_stats['accuracy']:.2%}")
                    
                    self._log_wandb({
                        "train/loss": avg_loss,
                        "train/accuracy": step_stats['accuracy'],
                        "train/mean_reward": step_stats['mean_reward'],
                    }, step=global_step)
                
                if global_step % CONFIG.grpo.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{global_step}")
            
            if is_main:
                print(f"   Epoch {epoch + 1} 完成: avg_loss={np.mean(epoch_losses):.4f}")
        
        self.save_checkpoint("final")
        self._finish_wandb()
        
        if is_main:
            print("\n" + "=" * 60)
            print(f"✅ {self.algorithm.upper()} 训练完成!")
            print("=" * 60)
    
    def save_checkpoint(self, name: str):
        """保存检查点"""
        if not self.accelerator.is_main_process:
            return
        
        save_path = self.output_dir / name
        save_path.mkdir(parents=True, exist_ok=True)
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        with open(save_path / "training_log.json", "w") as f:
            json.dump(self.training_log, f, indent=2)
        
        print(f"💾 保存到 {save_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("测试 VERL-style GRPO Trainer")
    print("=" * 60)
    
    trainer = VerlGRPOTrainer(algorithm="grpo")
    
    print("\n✅ 训练器创建成功")
    print(f"   算法: {trainer.algorithm}")
    print(f"   Advantage函数: {trainer.advantage_fn.__name__}")
    print(f"   Policy Loss函数: {trainer.policy_loss_fn.__name__}")
