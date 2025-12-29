# GRPO Math VERL 项目

## 项目特点

- **4卡分布式训练**：使用accelerate + DeepSpeed ZeRO-2
- **VERL核心算法**：GRPO/GSPO/RLOO + Dual-Clip PPO
- **多数据源支持**：GSM8K / OpenR1-Math / MATH
- **多格式Prompt**：根据数据源自动选择格式

## 数据源与Prompt格式

### 1. GSM8K (默认)
- **格式**：`#### [number]`
- **Prompt**：
```
Solve the following math problem step by step.

Problem: [question]

Please show your work and provide the final answer in the format: #### [answer]
```

### 2. OpenR1-Math 
- **格式**：`<answer>...</answer>`
- **Prompt**：
```
Solve the following math problem. Show your reasoning inside  tags, then give your final answer inside <answer></answer> tags.

Problem: [question]

[
```

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 本地数据（已包含）
- GSM8K: 7473条训练数据
- 如需使用OpenR1-Math：`python download_openr1_math.py --dataset openr1`

### 3. 运行训练
```bash
# GSM8K训练（默认）
python main.py --mode train --sources gsm8k

# OpenR1-Math训练
python main.py --mode train --sources openr1

# 4卡分布式训练
accelerate launch --config_file accelerate_config.yaml main.py --mode train --sources gsm8k
```

### 4. 算法选择
```bash
# GRPO (默认)
python main.py --mode train --algorithm grpo

# GSPO
python main.py --mode train --algorithm gspo

# RLOO
python main.py --mode train --algorithm rloo
```

## 配置说明

### 数据源配置
- `train_dataset`: gsm8k, openr1, math
- `openr1_sample_size`: 93722 (default split大小)
- `max_train_samples`: None (全量) 或指定数量

### 训练参数
- `num_epochs`: 3
- `batch_size`: 4 (per GPU)
- `learning_rate`: 1e-5
- `num_generations`: 4 (每个prompt生成候选数)

## 面试话术

"使用VERL框架重构了GRPO训练流程，核心算法采用VERL的组内标准化Advantage计算，支持GRPO/GSPO/RLOO三种算法切换。通过accelerate + DeepSpeed ZeRO-2实现4卡并行训练，7B模型可以高效训练。数据源支持GSM8K和OpenR1-Math，根据数据源自动选择Prompt格式。"