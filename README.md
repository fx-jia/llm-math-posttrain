# LLM Math Post-training

这是一个面向小型语言模型数学推理能力的后训练实验项目。项目以 GSM8K 为数据集，使用 LoRA 依次实验 SFT、DPO 和 GRPO/RLVR，并与未微调的基础模型进行对比。

## 项目目标

- 搭建可重复的数学推理后训练流程。
- 对比 Base、SFT、DPO 和 GRPO 的效果。
- 使用答案正确性、格式合规率、输出长度、推理延迟和显存占用进行评估。

## 整体流程

```text
GSM8K 原始数据
    ↓ prepare_gsm8k.py
训练集 / 测试集
    ├─→ train_sft_lora.py ─→ SFT LoRA
    │       ├─→ build_dpo_pairs.py ─→ train_dpo_lora.py ─→ DPO LoRA
    │       └─→ prepare_grpo_data.py ─→ train_grpo_lora.py ─→ GRPO LoRA
    └─→ eval_base.py / eval_lora.py ─→ 评测结果与对比分析
```

## 目录与文件说明

### `src/`

| 文件 | 功能 |
|---|---|
| `src/__init__.py` | 将 `src` 标记为 Python 包。 |
| `src/answer_utils.py` | 答案归一化工具。提取文本中最后的数值，消除货币符号、千分位、小数末尾零等表示差异，供评测脚本使用。 |
| `src/rewards.py` | GRPO 奖励函数。包含最终答案提取、正确性奖励、`Final Answer:` 格式奖励、过长输出惩罚及组合奖励。 |

### `scripts/`

#### 数据准备与检查

| 文件 | 功能 |
|---|---|
| `scripts/prepare_gsm8k.py` | 下载并转换 GSM8K，从原始答案中分离推理过程和最终答案，生成训练、测试 JSONL 文件，并校验输出。 |
| `scripts/inspect_gsm8k.py` | 统计处理后 GSM8K 的样本数、长度和答案格式，同时打印样例用于人工检查。 |
| `scripts/prepare_grpo_data.py` | 把 GSM8K 训练数据转成 GRPOTrainer 需要的 `prompt` 和 `answer` 格式。 |
| `scripts/build_dpo_pairs.py` | 用 SFT 模型采样回答，保留格式正确但答案错误的结果作为 rejected，与标准推理 chosen 组成 DPO 偏好对。 |
| `scripts/inspect_dpo_pairs.py` | 检查 DPO 偏好对数量、chosen/rejected 长度、格式合规率和样例内容。 |

#### 模型训练

| 文件 | 功能 |
|---|---|
| `scripts/train_sft_lora.py` | 使用标准推理文本进行 LoRA 监督微调，适配注意力层和 MLP 层。 |
| `scripts/train_dpo_lora.py` | 从 SFT 适配器启动 DPO 训练。分别加载可训练的策略模型和冻结的参考模型，学习 chosen/rejected 偏好。 |
| `scripts/train_grpo_lora.py` | 从 SFT 适配器启动 GRPO 训练，用答案正确性、输出格式和长度作为可验证奖励。 |
| `scripts/check_lora_adapter.py` | 检查 LoRA 输出目录是否包含配置和权重，并打印 rank、alpha、目标层及权重大小。 |

#### 评测与分析

| 文件 | 功能 |
|---|---|
| `scripts/smoke_test_model.py` | 加载基础模型并完成一道数学题的生成，用于快速检查环境、模型和 GPU。 |
| `scripts/eval_base.py` | 在 GSM8K 测试集上评估未微调模型，记录 Exact Match、格式合规率、延迟、输出 token 数和完整生成内容。 |
| `scripts/eval_lora.py` | 加载指定 LoRA 适配器进行与基础模型相同的评测。 |
| `scripts/summarize_eval.py` | 汇总一个或多个评测 JSONL，以 Markdown 表格输出样本数、准确率、格式率、延迟和平均长度。 |
| `scripts/recompute_eval_metrics.py` | 使用当前答案归一化规则重算历史评测文件的指标，用于发现解析规则更新造成的指标变化。 |
| `scripts/analyze_eval_diff.py` | 按样本 ID 比较两份评测结果，划分为均正确、A 对 B 错、A 错 B 对和均错误，并展示典型样例。 |
| `scripts/test_answer_normalization.py` | 用若干边界样例检查答案归一化逻辑。 |
| `scripts/test_grpo_rewards.py` | 手工构造 completion，检查 GRPO 答案提取与各项奖励计算。 |

### 数据、输出与记录

| 路径 | 功能 |
|---|---|
| `data/processed/gsm8k_train.jsonl` | 处理后的 GSM8K 训练集，包含题目、推理、答案和 SFT 文本。 |
| `data/processed/gsm8k_test.jsonl` | 处理后的 GSM8K 测试集。 |
| `data/processed/dpo_pairs_sft.jsonl` | 由 SFT 模型采样构建的 DPO 偏好对。 |
| `data/processed/grpo_train.jsonl` | GRPO 训练数据，主要包含 prompt 和可验证答案。 |
| `outputs/<run_name>/` | 训练产生的 LoRA 适配器、tokenizer 和训练配置。 |
| `outputs/eval_*.jsonl` | 逐样本评测结果。 |
| `logs/<run_name>/train.log` | 各次训练的终端日志。 |
| `notes/experiment_log.md` | 实验命令、配置和结果记录。 |
| `notes/report_v1.md` | 第一版实验报告与结果分析。 |

`__pycache__/`、`.DS_Store` 和 `.git/` 是 Python、macOS 和 Git 自动生成的辅助文件，不属于项目业务逻辑。

## 常用命令

```bash
# 1. 准备并检查数据
python scripts/prepare_gsm8k.py
python scripts/inspect_gsm8k.py

# 2. 进行小规模 SFT 烟雾训练
python scripts/train_sft_lora.py --train-limit 256 --output-dir outputs/sft_smoke

# 3. 评估基础模型与 LoRA 模型
python scripts/eval_base.py --limit 20
python scripts/eval_lora.py --adapter-dir outputs/sft_smoke --limit 20

# 4. 汇总评测结果
python scripts/summarize_eval.py outputs/eval_base_limit20.jsonl outputs/eval_sft_smoke_limit20.jsonl

# 5. 运行轻量逻辑测试
python scripts/test_answer_normalization.py
python scripts/test_grpo_rewards.py
```

## 默认模型与配置

默认基础模型为 `Qwen/Qwen2.5-1.5B`，可通过环境变量替换：

```bash
MODEL_NAME=/path/to/model python scripts/eval_base.py --limit 20
SFT_ADAPTER_DIR=/path/to/sft_adapter python scripts/train_dpo_lora.py
```

训练脚本默认使用 BF16 和自动设备映射，因此实际运行前需确保当前硬件和 PyTorch 环境支持相应配置。
