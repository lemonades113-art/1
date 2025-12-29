# -*- coding: utf-8 -*-
"""
OpenR1-Math 数据下载脚本 (VERL版)
==================================

支持两种下载方式：
1. 直接下载 default split (93.7k条，预采样版本，推荐!)
2. 流式下载 + 实时抽样 (节省内存)

使用HF镜像加速下载

用法：
    # 方式1：下载预采样版本 (推荐，最快)
    python download_openr1_math.py --method default --sample_size 30000
    
    # 方式2：流式下载
    python download_openr1_math.py --method streaming --sample_size 30000
"""

import os
import json
import random
from pathlib import Path
from typing import Optional
import argparse

# 设置HF镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def download_default_split(
    sample_size: int = 30000,
    output_dir: str = "./data/openr1_math"
) -> Optional[str]:
    """
    下载 OpenR1-Math-220k 的 default split
    
    Default split 只有 93,722 条，是预采样版本！
    比完整的 220k 版本下载更快
    """
    try:
        from datasets import load_dataset
        print("📥 下载 OpenR1-Math-220k (default split: 93.7k条)...")
        print("   使用镜像: https://hf-mirror.com")
        
        # 下载 default split
        dataset = load_dataset(
            "open-r1/OpenR1-Math-220k",
            split="default"  # 预采样版本
        )
        print(f"✅ 下载完成，总条数: {len(dataset)}")
        
        # 抽样
        if sample_size < len(dataset):
            print(f"🎲 随机抽样 {sample_size} 条...")
            indices = random.sample(range(len(dataset)), sample_size)
            sampled = dataset.select(indices)
        else:
            sampled = dataset
            sample_size = len(dataset)
        
        # 保存
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        data = []
        for item in sampled:
            data.append({
                "problem": item.get("problem", ""),
                "solution": item.get("solution", ""),
                "answer": item.get("answer", ""),
                "source": "openr1_math_default"
            })
        
        output_file = output_path / f"train_{sample_size}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 保存到 {output_file}")
        print(f"   样本数: {len(data)}")
        
        return str(output_file)
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


def download_streaming(
    sample_size: int = 30000,
    output_dir: str = "./data/openr1_math"
) -> Optional[str]:
    """
    流式下载 + 蓄水池抽样
    
    边下载边抽样，内存占用极低
    适合大数据集场景
    """
    try:
        from datasets import load_dataset
        print("📥 流式下载 OpenR1-Math-220k...")
        print("   使用镜像 + 流式模式，节省内存")
        
        # 流式加载
        dataset = load_dataset(
            "open-r1/OpenR1-Math-220k",
            split="default",
            streaming=True  # 流式模式
        )
        
        # 蓄水池抽样
        print(f"🎲 蓄水池抽样 {sample_size} 条...")
        reservoir = []
        count = 0
        
        for item in dataset:
            count += 1
            if count <= sample_size:
                reservoir.append({
                    "problem": item.get("problem", ""),
                    "solution": item.get("solution", ""),
                    "answer": item.get("answer", ""),
                    "source": "openr1_math_streaming"
                })
            else:
                # 蓄水池替换
                j = random.randint(0, count - 1)
                if j < sample_size:
                    reservoir[j] = {
                        "problem": item.get("problem", ""),
                        "solution": item.get("solution", ""),
                        "answer": item.get("answer", ""),
                        "source": "openr1_math_streaming"
                    }
            
            if count % 10000 == 0:
                print(f"   已处理: {count} 条")
        
        print(f"✅ 处理完成，总条数: {count}")
        
        # 保存
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"train_{len(reservoir)}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(reservoir, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 保存到 {output_file}")
        print(f"   样本数: {len(reservoir)}")
        
        return str(output_file)
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_gsm8k(output_dir: str = "./data/gsm8k") -> Optional[str]:
    """下载 GSM8K 数据集"""
    try:
        from datasets import load_dataset
        print("📥 下载 GSM8K...")
        
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        print(f"✅ GSM8K 总条数: {len(dataset)}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        data = []
        for item in dataset:
            answer = item["answer"].split("####")[-1].strip() if "####" in item["answer"] else ""
            data.append({
                "problem": item["question"],
                "solution": item["answer"],
                "answer": answer,
                "source": "gsm8k"
            })
        
        output_file = output_path / f"train_{len(data)}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 保存到 {output_file}")
        return str(output_file)
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="下载数学数据集")
    parser.add_argument(
        "--dataset",
        choices=["openr1", "gsm8k"],
        default="openr1",
        help="数据集选择"
    )
    parser.add_argument(
        "--method",
        choices=["default", "streaming"],
        default="default",
        help="下载方式：default (推荐) 或 streaming"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=30000,
        help="抽样数量"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/openr1_math",
        help="输出目录"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("OpenR1-Math 数据下载脚本")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"方式: {args.method}")
    print(f"抽样数: {args.sample_size}")
    print(f"输出: {args.output_dir}")
    print("=" * 60)
    
    if args.dataset == "openr1":
        if args.method == "default":
            download_default_split(args.sample_size, args.output_dir)
        else:
            download_streaming(args.sample_size, args.output_dir)
    else:
        download_gsm8k(args.output_dir)


if __name__ == "__main__":
    main()
