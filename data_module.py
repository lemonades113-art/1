# -*- coding: utf-8 -*-
"""
GRPO Math 数据模块 (VERL风格)
==============================

支持数据集：
- GSM8K: 8500道小学数学题
- MATH: 12500道高中/大学数学题
- OpenR1-Math-220k: 22万推理问题 (含<think><answer>格式)

特性：
- 流式下载 + HF镜像
- 自动抽样
- 分布式数据加载

来源: 复制自原grpo_math项目，添加OpenR1支持
"""

import os
import re
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# 设置HF镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    from datasets import load_dataset, Dataset, IterableDataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️ datasets库未安装，将使用模拟数据")

from config import CONFIG


@dataclass
class MathProblem:
    """数学问题数据结构"""
    problem_id: str
    question: str
    answer: str  # 标准答案
    solution: str  # 完整解答过程
    difficulty: str  # easy, medium, hard
    source: str  # gsm8k, math, openr1
    category: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "problem_id": self.problem_id,
            "question": self.question,
            "answer": self.answer,
            "solution": self.solution,
            "difficulty": self.difficulty,
            "source": self.source,
            "category": self.category,
        }
    
    def get_prompt(self) -> str:
        """生成训练用prompt，根据数据源自动选择格式"""
        if self.source == "openr1":
            # OpenR1格式：使用<think><answer>标签
            return f"""Solve the following math problem. Show your reasoning inside <think></think> tags, then give your final answer inside <answer></answer> tags.

Problem: {self.question}

<think>
"""
        else:
            # GSM8K/MATH格式：使用#### number
            return f"""Solve the following math problem step by step.

Problem: {self.question}

Please show your work and provide the final answer in the format: #### [answer]"""


class AnswerExtractor:
    """答案提取器 (复制自原项目)"""
    
    @staticmethod
    def extract_gsm8k_answer(text: str) -> Optional[str]:
        pattern = r'####\s*([-+]?\d*\.?\d+)'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return None
    
    @staticmethod
    def extract_math_answer(text: str) -> Optional[str]:
        pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return None
    
    @staticmethod
    def extract_openr1_answer(text: str) -> Optional[str]:
        """提取OpenR1格式答案 (<answer>...</answer>)"""
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            # 尝试从answer中提取数字
            num_match = re.search(r'([-+]?\d*\.?\d+)', answer_text)
            if num_match:
                return num_match.group(1)
            return answer_text
        return None
    
    @staticmethod
    def extract_number(text: str) -> Optional[str]:
        pattern = r'([-+]?\d*\.?\d+)'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
        return None
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        if answer is None:
            return ""
        answer = answer.strip()
        
        # 处理分数
        if '/' in answer:
            try:
                parts = answer.split('/')
                if len(parts) == 2:
                    num = float(parts[0])
                    den = float(parts[1])
                    answer = str(num / den)
            except:
                pass
        
        # 四舍五入
        try:
            answer = str(round(float(answer), 4))
        except:
            pass
        
        return answer
    
    @classmethod
    def extract(cls, text: str, source: str = "gsm8k") -> Optional[str]:
        if source == "gsm8k":
            answer = cls.extract_gsm8k_answer(text)
        elif source == "math":
            answer = cls.extract_math_answer(text)
        elif source == "openr1":
            answer = cls.extract_openr1_answer(text)
        else:
            answer = cls.extract_number(text)
        
        if answer is None:
            answer = cls.extract_number(text)
        
        return cls.normalize_answer(answer) if answer else None


class GSM8KLoader:
    """GSM8K数据加载器 (支持本地文件和在线下载)"""
    
    # 本地数据文件路径
    LOCAL_DATA_PATHS = [
        "./data/gsm8k/train_full_7473.json",
        "../grpo_math/data/gsm8k/train_full_7473.json",
        "./data/gsm8k/train.json",
    ]
    
    @classmethod
    def load(cls, split: str = "train", max_samples: int = None) -> List[MathProblem]:
        # 优先从本地加载
        local_data = cls._load_local(max_samples)
        if local_data:
            return local_data
        
        # 本地没有则在线下载
        if not HAS_DATASETS:
            return cls._load_mock(split, max_samples)
        
        try:
            print(f"📥 在线下载 GSM8K (使用镜像 {os.environ.get('HF_ENDPOINT', 'default')})...")
            dataset = load_dataset(
                "gsm8k", "main",
                split=split,
                cache_dir=CONFIG.data.cache_dir
            )
            
            problems = []
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                
                answer = AnswerExtractor.extract(item["answer"], source="gsm8k")
                problem = MathProblem(
                    problem_id=f"gsm8k_{split}_{i}",
                    question=item["question"],
                    answer=answer or "",
                    solution=item["answer"],
                    difficulty=cls._estimate_difficulty(item["answer"]),
                    source="gsm8k"
                )
                problems.append(problem)
            
            print(f"✅ 在线加载GSM8K {split}: {len(problems)} 条")
            return problems
            
        except Exception as e:
            print(f"⚠️ 加载GSM8K失败: {e}，使用模拟数据")
            return cls._load_mock(split, max_samples)
    
    @classmethod
    def _load_local(cls, max_samples: int = None) -> Optional[List[MathProblem]]:
        """从本地JSON文件加载"""
        for path in cls.LOCAL_DATA_PATHS:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    problems = []
                    for i, item in enumerate(data):
                        if max_samples and i >= max_samples:
                            break
                        
                        # 兼容多种格式
                        question = item.get("question", item.get("problem", ""))
                        solution = item.get("answer", item.get("solution", ""))
                        answer = AnswerExtractor.extract(solution, source="gsm8k")
                        
                        problem = MathProblem(
                            problem_id=f"gsm8k_local_{i}",
                            question=question,
                            answer=answer or "",
                            solution=solution,
                            difficulty=cls._estimate_difficulty(solution),
                            source="gsm8k"
                        )
                        problems.append(problem)
                    
                    print(f"✅ 从本地加载GSM8K: {path} ({len(problems)} 条)")
                    return problems
                    
                except Exception as e:
                    print(f"⚠️ 加载本地文件失败 {path}: {e}")
                    continue
        
        return None  # 本地没有数据
    
    @classmethod
    def _estimate_difficulty(cls, solution: str) -> str:
        steps = len(re.split(r'[.\n]', solution))
        if steps <= 3:
            return "easy"
        elif steps <= 6:
            return "medium"
        else:
            return "hard"
    
    @classmethod
    def _load_mock(cls, split: str, max_samples: int = None) -> List[MathProblem]:
        mock_problems = [
            {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remainder for $2 each. How much does she make?",
             "answer": "18", "solution": "16 - 3 - 4 = 9 eggs. 9 * 2 = $18. #### 18"},
            {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?",
             "answer": "3", "solution": "White: 2/2 = 1. Total: 2 + 1 = 3. #### 3"},
        ]
        
        problems = []
        n = max_samples or len(mock_problems)
        for i in range(min(n, len(mock_problems))):
            item = mock_problems[i % len(mock_problems)]
            problems.append(MathProblem(
                problem_id=f"gsm8k_mock_{i}",
                question=item["question"],
                answer=item["answer"],
                solution=item["solution"],
                difficulty="medium",
                source="gsm8k"
            ))
        
        print(f"📦 使用GSM8K模拟数据: {len(problems)} 条")
        return problems


class OpenR1MathLoader:
    """
    OpenR1-Math-220k 数据加载器
    
    数据集: https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
    
    特点：
    - default split: 93.7k条 (SFT优化后)
    - extended split: 131k条
    - 包含 <think>...<answer> 格式的推理轨迹
    
    支持：
    - 流式下载 (streaming=True)
    - HF镜像站
    - 自动抽样
    """
    
    @classmethod
    def load(
        cls,
        split: str = "train",
        max_samples: int = None,
        streaming: bool = True
    ) -> List[MathProblem]:
        """
        加载OpenR1-Math数据
        
        Args:
            split: "train" (default 93.7k) 或 "train[extended]" (131k)
            max_samples: 最大样本数，None表示全量
            streaming: 是否使用流式加载
        """
        if not HAS_DATASETS:
            return cls._load_mock(max_samples)
        
        try:
            dataset_name = CONFIG.data.openr1_path
            sample_size = max_samples or CONFIG.data.openr1_sample_size
            
            print(f"📥 加载 OpenR1-Math-220k (streaming={streaming}, max={sample_size})...")
            
            if streaming:
                # 流式加载：边下载边处理，节省内存
                dataset = load_dataset(
                    dataset_name,
                    split="train",
                    streaming=True,
                    cache_dir=CONFIG.data.cache_dir
                )
                
                problems = []
                for i, item in enumerate(dataset):
                    if sample_size and len(problems) >= sample_size:
                        break
                    
                    problem = cls._parse_item(item, i)
                    if problem:
                        problems.append(problem)
                    
                    if (i + 1) % 5000 == 0:
                        print(f"   已处理 {i + 1} 条，已采样 {len(problems)} 条")
            else:
                # 非流式：一次性加载
                dataset = load_dataset(
                    dataset_name,
                    split="train",
                    cache_dir=CONFIG.data.cache_dir
                )
                
                # 随机抽样
                if sample_size and sample_size < len(dataset):
                    indices = random.sample(range(len(dataset)), sample_size)
                    dataset = dataset.select(indices)
                
                problems = []
                for i, item in enumerate(dataset):
                    problem = cls._parse_item(item, i)
                    if problem:
                        problems.append(problem)
            
            print(f"✅ 加载OpenR1-Math: {len(problems)} 条")
            return problems
            
        except Exception as e:
            print(f"⚠️ 加载OpenR1-Math失败: {e}")
            import traceback
            traceback.print_exc()
            return cls._load_mock(max_samples)
    
    @classmethod
    def _parse_item(cls, item: Dict, idx: int) -> Optional[MathProblem]:
        """解析单条数据"""
        try:
            problem_text = item.get("problem", "")
            solution_text = item.get("solution", "")
            answer_text = item.get("answer", "")
            
            if not problem_text:
                return None
            
            # 提取答案
            answer = answer_text if answer_text else AnswerExtractor.extract(solution_text, "openr1")
            
            return MathProblem(
                problem_id=f"openr1_{idx}",
                question=problem_text,
                answer=answer or "",
                solution=solution_text,
                difficulty="medium",  # OpenR1没有难度标签
                source="openr1"
            )
        except Exception as e:
            return None
    
    @classmethod
    def _load_mock(cls, max_samples: int = None) -> List[MathProblem]:
        """模拟数据"""
        mock = [
            {"question": "What is 2 + 2?", "answer": "4",
             "solution": "<think>\nI need to add 2 and 2.\n2 + 2 = 4\n</think>\n<answer>\n4\n</answer>"},
        ]
        
        problems = []
        n = max_samples or 10
        for i in range(min(n, len(mock))):
            item = mock[i % len(mock)]
            problems.append(MathProblem(
                problem_id=f"openr1_mock_{i}",
                question=item["question"],
                answer=item["answer"],
                solution=item["solution"],
                difficulty="medium",
                source="openr1"
            ))
        
        print(f"📦 使用OpenR1模拟数据: {len(problems)} 条")
        return problems


class MathDataset:
    """数学数据集管理器"""
    
    def __init__(
        self,
        sources: List[str] = None,
        max_train: int = None,
        max_eval: int = None
    ):
        self.sources = sources or ["gsm8k"]
        self.max_train = max_train or CONFIG.data.max_train_samples
        self.max_eval = max_eval or CONFIG.data.max_eval_samples
        
        self.train_data: List[MathProblem] = []
        self.eval_data: List[MathProblem] = []
    
    def load(self):
        """加载数据"""
        for source in self.sources:
            if source == "gsm8k":
                train = GSM8KLoader.load("train", self.max_train)
                self.train_data.extend(train[:-500] if len(train) > 500 else train)
                self.eval_data.extend(train[-500:] if len(train) > 500 else [])
                
            elif source == "openr1":
                train = OpenR1MathLoader.load(
                    max_samples=self.max_train or CONFIG.data.openr1_sample_size,
                    streaming=True
                )
                split_idx = int(len(train) * 0.95)
                self.train_data.extend(train[:split_idx])
                self.eval_data.extend(train[split_idx:])
        
        random.shuffle(self.train_data)
        
        print(f"\n📊 数据集统计:")
        print(f"   训练集: {len(self.train_data)}")
        print(f"   验证集: {len(self.eval_data)}")
    
    def get_train_dataset(self) -> List[Dict]:
        """获取训练数据"""
        return [
            {
                "prompt": p.get_prompt(),
                "answer": p.answer,
                "solution": p.solution,
                "problem_id": p.problem_id,
            }
            for p in self.train_data
        ]
    
    def get_eval_dataset(self) -> List[Dict]:
        """获取评估数据"""
        return [p.to_dict() for p in self.eval_data]


# ============================================================================
# 分布式数据加载支持 (VERL风格)
# ============================================================================

def create_distributed_dataloader(
    dataset: List[Dict],
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool = True,
    drop_last: bool = True
):
    """
    创建分布式数据加载器
    
    每个GPU只加载自己负责的那部分数据
    """
    # 按rank切分数据
    total = len(dataset)
    per_rank = total // world_size
    start = rank * per_rank
    end = start + per_rank if rank < world_size - 1 else total
    
    local_dataset = dataset[start:end]
    
    if shuffle:
        random.shuffle(local_dataset)
    
    # 生成batch
    batches = []
    for i in range(0, len(local_dataset), batch_size):
        batch = local_dataset[i:i + batch_size]
        if drop_last and len(batch) < batch_size:
            continue
        batches.append(batch)
    
    return batches


if __name__ == "__main__":
    print("=" * 60)
    print("测试数据模块")
    print("=" * 60)
    
    # 测试GSM8K
    dataset = MathDataset(sources=["gsm8k"], max_train=100, max_eval=20)
    dataset.load()
    
    if dataset.train_data:
        p = dataset.train_data[0]
        print(f"\n📝 GSM8K样例:")
        print(f"  问题: {p.question[:80]}...")
        print(f"  答案: {p.answer}")
    
    # 测试答案提取
    print("\n🔍 答案提取测试:")
    test_cases = [
        ("#### 42", "gsm8k"),
        ("\\boxed{3.14}", "math"),
        ("<answer>\n100\n</answer>", "openr1"),
    ]
    for text, source in test_cases:
        answer = AnswerExtractor.extract(text, source)
        print(f"  [{source}] '{text[:20]}...' → {answer}")
