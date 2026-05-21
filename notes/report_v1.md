# 数学推理大模型后训练实验报告 V1

## 1. 项目目标

本项目面向 GSM8K 数学推理任务，构建一个小规模但完整的大模型后训练流程，覆盖 Base 评估、LoRA SFT、DPO、GRPO/RLVR、自动评估与错误分析。

选择数学推理任务的原因是：最终答案可验证，适合做自动评估和基于规则奖励的强化学习后训练。

## 2. 实验环境

- 平台：AutoDL
- GPU：NVIDIA RTX 5090 32GB
- 基座模型：Qwen/Qwen2.5-1.5B
- 框架：PyTorch、Transformers、PEFT、TRL、Datasets
- 精度：BF16
- 微调方式：LoRA

## 3. 数据集

数据集：GSM8K

- train：7473 条
- test：1319 条
- 当前评估子集：test 前 100 条

预处理方式：

- 从 GSM8K answer 字段中提取 reasoning 和 final answer
- 统一构造 SFT 输入格式
- 要求模型最终输出包含 Final Answer:

## 4. 方法

### 4.1 Base 评估

使用 Qwen2.5-1.5B 原始模型进行 greedy decoding，作为后续 SFT、DPO、GRPO 的 baseline。

### 4.2 LoRA SFT

SFT 阶段使用 GSM8K 标准推理过程监督微调模型，让模型学习稳定的数学推理格式。

LoRA 配置：

- rank：8
- alpha：16
- dropout：0.05
- 可训练参数：约 9.23M
- 可训练参数占比：0.5945%
- max length：512
- epoch：1
- learning rate：2e-4

### 4.3 DPO

DPO 阶段使用自动构造的 chosen/rejected 偏好对。

构造方式：

- 用 SFT 模型对训练题采样生成答案
- 保留最终答案错误但格式合规的输出作为 rejected
- 使用 GSM8K 标准解答作为 chosen

DPO smoke 配置：

- 偏好对数量：200
- beta：0.1
- learning rate：5e-6
- epoch：1

说明：这不是人工 RLHF，而是基于可验证答案自动构造的偏好优化数据。

### 4.4 GRPO / RLVR

GRPO 阶段使用规则奖励函数进行强化学习式后训练。

奖励设计：

- 答案正确：+1.0
- 包含 Final Answer:：+0.1
- 缺少 Final Answer:：-0.2
- 输出过长：-0.1

GRPO smoke 配置：

- 训练样本：64
- max steps：20
- num generations：4
- learning rate：1e-6
- beta：0.01

## 5. 当前实验结果

评估集：GSM8K test 前 100 条。

| Model | Exact Match | Format Rate | Avg Latency | Avg Output Tokens |
|---|---:|---:|---:|---:|
| Base Qwen2.5-1.5B | 0.5100 | 0.8800 | 2.38s | 181.7 |
| SFT LoRA r=8 | 0.6800 | 0.9700 | 3.70s | 129.1 |
| DPO smoke | 0.6900 | 0.9900 | 3.45s | 120.0 |
| GRPO smoke | 0.6200 | 0.9800 | 3.60s | 126.4 |

## 6. 主要结论

SFT 是当前主要收益来源，使 Exact Match 从 0.5100 提升到 0.6800，提升 17 个百分点。

DPO smoke 在 200 条偏好对上进一步将 Exact Match 提升到 0.6900，同时格式合规率提升到 0.9900。

GRPO smoke 跑通了 RLVR 训练链路，但当前 20 step 小规模训练使 Exact Match 下降到 0.6200，说明 reward 设计、采样方差和训练步数还需要进一步优化。

## 7. 错误分析与评估修正

错误分析中发现，原始字符串匹配会把数值等价答案误判为错误，例如：

- 26.00 vs 26
- 88.00 vs 88

因此新增了统一的数值归一化逻辑，并重新生成了评估结果。

这个修正说明：数学推理评估不能只做字符串匹配，必须考虑数值等价性。

## 8. 当前局限

- 当前只评估了 GSM8K test 前 100 条
- DPO 只用了 200 条偏好对
- GRPO 只训练了 20 step
- GRPO 中部分 batch 的 reward_std 为 0，有效学习信号不足
- reward 只验证最终答案，没有检查中间推理过程
- LoRA adapter 尚未 merge，推理延迟可能偏高

## 9. 下一步计划

- 在完整 GSM8K test set 上评估 Base、SFT、DPO、GRPO
- 扩大 DPO 偏好数据规模
- 优化 GRPO reward 和训练步数
- 对比 LoRA rank 8 和 rank 16
- 搭建 Gradio Demo 展示模型推理过程
