# -*- coding: utf-8 -*-
"""
数学推理 Reward 函数 (VERL风格)
================================

参考 open-r1 和原 grpo_math 项目实现
支持 GSM8K 和 OpenR1-Math 格式

面试话术：
"Reward函数支持多种答案格式提取，包括GSM8K的#### number格式
和OpenR1的<answer>标签格式。支持正确性验证、格式奖励和长度控制。"
"""

import re
from typing import Optional, Tuple, List
from dataclasses import dataclass

from config import CONFIG


@dataclass
class RewardResult:
    """Reward计算结果"""
    is_correct: bool = False
    correctness: float = 0.0
    format_score: float = 0.0
    length_score: float = 0.0
    total: float = 0.0
    
    extracted_answer: str = ""
    gold_answer: str = ""


class MathRewardFunction:
    """
    数学推理 Reward 函数
    
    支持多种答案格式：
    1. GSM8K: #### number
    2. OpenR1: <answer>...</answer>
    3. MATH: \\boxed{...}
    
    面试话术：
    "Reward设计包括正确性(核心)、格式规范、长度控制三个维度。
    正确性验证支持数值容差匹配，避免浮点精度问题。"
    """
    
    def __init__(
        self,
        correctness_weight: float = None,
        format_weight: float = None,
        length_weight: float = None,
        epsilon: float = 1e-5,
    ):
        self.correctness_weight = correctness_weight or CONFIG.reward.correctness_weight
        self.format_weight = format_weight or CONFIG.reward.format_weight
        self.length_weight = length_weight or CONFIG.reward.length_weight
        self.epsilon = epsilon
    
    def compute(self, response: str, gold_answer: str, source: str = "gsm8k") -> RewardResult:
        """
        计算 Reward
        
        Args:
            response: 模型生成的响应
            gold_answer: 标准答案
            source: 数据源 (gsm8k, openr1, math)
            
        Returns:
            RewardResult
        """
        result = RewardResult(gold_answer=gold_answer)
        
        # 1. 提取答案
        extracted = self._extract_answer(response, source)
        result.extracted_answer = extracted or ""
        
        # 2. 正确性检查
        if extracted and gold_answer:
            result.is_correct = self._check_correctness(extracted, gold_answer)
            result.correctness = 1.0 if result.is_correct else 0.0
        
        # 3. 格式检查
        result.format_score = self._check_format(response, source)
        
        # 4. 长度检查
        result.length_score = self._check_length(response)
        
        # 5. 计算总分
        result.total = (
            result.correctness * self.correctness_weight +
            result.format_score * self.format_weight +
            result.length_score * self.length_weight
        )
        
        return result
    
    def _extract_answer(self, text: str, source: str = "gsm8k") -> Optional[str]:
        """从响应中提取答案"""
        
        # 尝试 <answer> 标签 (OpenR1格式)
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            # 从答案中提取数字
            num_match = re.search(r'([-+]?\d*\.?\d+)', answer)
            if num_match:
                return self._normalize_number(num_match.group(1))
            return answer
        
        # 尝试 #### 格式 (GSM8K格式)
        gsm8k_match = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
        if gsm8k_match:
            return self._normalize_number(gsm8k_match.group(1))
        
        # 尝试 \boxed{} 格式 (MATH格式)
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            content = boxed_match.group(1).strip()
            num_match = re.search(r'([-+]?\d*\.?\d+)', content)
            if num_match:
                return self._normalize_number(num_match.group(1))
            return content
        
        # 最后尝试提取最后一个数字
        numbers = re.findall(r'([-+]?\d*\.?\d+)', text)
        if numbers:
            return self._normalize_number(numbers[-1])
        
        return None
    
    def _normalize_number(self, num_str: str) -> str:
        """标准化数字"""
        try:
            num = float(num_str)
            # 如果是整数，去掉小数点
            if num == int(num):
                return str(int(num))
            # 四舍五入到4位小数
            return str(round(num, 4))
        except:
            return num_str.strip()
    
    def _check_correctness(self, extracted: str, gold: str) -> bool:
        """检查答案正确性"""
        # 标准化两个答案
        extracted_norm = self._normalize_number(extracted)
        gold_norm = self._normalize_number(gold)
        
        # 直接字符串匹配
        if extracted_norm == gold_norm:
            return True
        
        # 数值匹配（带容差）
        try:
            ext_num = float(extracted_norm)
            gold_num = float(gold_norm)
            
            # 绝对误差
            if abs(ext_num - gold_num) < self.epsilon:
                return True
            
            # 相对误差
            if abs(gold_num) > self.epsilon:
                if abs(ext_num - gold_num) / abs(gold_num) < self.epsilon:
                    return True
        except:
            pass
        
        return False
    
    def _check_format(self, response: str, source: str = "gsm8k") -> float:
        """
        检查格式规范性
        
        与 open-r1/rewards.py 的 tag_count_reward 一致
        """
        score = 0.0
        
        if source == "openr1":
            # OpenR1 格式：严格匹配 <think>\n...\n</think>\n<answer>\n...\n</answer>
            # 参考 open-r1/src/open_r1/rewards.py L99-112 的 tag_count_reward
            if response.count("<think>\n") == 1:
                score += 0.25
            if response.count("\n</think>\n") == 1:
                score += 0.25
            if response.count("\n<answer>\n") == 1:
                score += 0.25
            if response.count("\n</answer>") == 1:
                score += 0.25
        else:
            # GSM8K 格式：推理步骤 + #### answer
            # 参考 open-r1/src/open_r1/rewards.py L115-129 的 reasoning_steps_reward
            steps_pattern = r'(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)'
            steps = len(re.findall(steps_pattern, response))
            if steps >= 3:
                score += 0.5
            elif steps >= 1:
                score += min(0.5, steps / 3)
            
            # 检查是否有 ####
            if '####' in response:
                score += 0.5
        
        return min(1.0, score)
    
    def _check_length(self, response: str) -> float:
        """检查长度合理性"""
        length = len(response)
        min_len = CONFIG.reward.min_length
        max_len = CONFIG.reward.max_length
        optimal_len = CONFIG.reward.optimal_length
        
        if length < min_len:
            # 太短
            return max(0, length / min_len)
        elif length <= optimal_len:
            # 在理想范围内
            return 1.0
        elif length <= max_len:
            # 有点长但还行
            return 1.0 - 0.5 * (length - optimal_len) / (max_len - optimal_len)
        else:
            # 太长
            return 0.5 - 0.5 * min(1.0, (length - max_len) / max_len)


class RewardTracker:
    """
    Reward 统计追踪器
    
    追踪训练过程中的准确率和reward统计
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct_count = 0
        self.total_count = 0
        self.total_reward = 0.0
        self.rewards = []
    
    def add(self, result: RewardResult):
        """添加一条记录"""
        self.total_count += 1
        if result.is_correct:
            self.correct_count += 1
        self.total_reward += result.total
        self.rewards.append(result.total)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        if self.total_count == 0:
            return {
                "accuracy": 0.0,
                "mean_reward": 0.0,
                "total_count": 0,
            }
        
        return {
            "accuracy": self.correct_count / self.total_count,
            "mean_reward": self.total_reward / self.total_count,
            "total_count": self.total_count,
            "correct_count": self.correct_count,
        }


# ============================================================================
# 兼容 open-r1 的 reward 函数接口
# ============================================================================

def accuracy_reward(
    completions: List[List[dict]], 
    solution: List[str], 
    **kwargs
) -> List[Optional[float]]:
    """
    兼容 open-r1 的准确性 reward 函数
    
    Args:
        completions: 模型输出，格式 [[{"content": "..."}], ...]
        solution: 标准答案列表
    
    Returns:
        rewards: 每个样本的reward
    """
    reward_fn = MathRewardFunction()
    rewards = []
    
    for completion, sol in zip(completions, solution):
        content = completion[0]["content"] if completion else ""
        result = reward_fn.compute(content, sol, source="openr1")
        rewards.append(result.correctness if result.is_correct else 0.0)
    
    return rewards


def format_reward(completions: List[List[dict]], **kwargs) -> List[float]:
    """
    格式 reward 函数
    
    检查是否有 <think>...</think><answer>...</answer> 格式
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
        rewards.append(1.0 if match else 0.0)
    
    return rewards


def tag_count_reward(completions: List[List[dict]], **kwargs) -> List[float]:
    """
    标签计数 reward
    
    分别检查 <think>, </think>, <answer>, </answer> 各0.25分
    """
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count
    
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        rewards.append(count_tags(content))
    
    return rewards


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 MathRewardFunction")
    print("=" * 60)
    
    reward_fn = MathRewardFunction()
    
    # 测试用例
    test_cases = [
        # (response, gold_answer, source, expected_correct)
        ("Let me solve this step by step.\n\n2 + 2 = 4\n\n#### 4", "4", "gsm8k", True),
        ("<think>\nI need to calculate 2 + 2.\n2 + 2 = 4\n</think>\n<answer>\n4\n</answer>", "4", "openr1", True),
        ("The answer is \\boxed{42}", "42", "math", True),
        ("#### 5", "4", "gsm8k", False),
        ("<answer>3.14</answer>", "3.14", "openr1", True),
    ]
    
    print("\n测试答案提取和验证:")
    for response, gold, source, expected in test_cases:
        result = reward_fn.compute(response, gold, source)
        status = "✅" if result.is_correct == expected else "❌"
        print(f"{status} [{source}] extracted='{result.extracted_answer}', gold='{gold}', correct={result.is_correct}")
        print(f"   total_reward={result.total:.4f}")
    
    # 测试 Tracker
    print("\n测试 RewardTracker:")
    tracker = RewardTracker()
    for response, gold, source, _ in test_cases:
        result = reward_fn.compute(response, gold, source)
        tracker.add(result)
    
    stats = tracker.get_stats()
    print(f"   Accuracy: {stats['accuracy']:.2%}")
    print(f"   Mean Reward: {stats['mean_reward']:.4f}")
    print(f"   Total: {stats['total_count']}")
    
    print("\n✅ 测试完成!")
