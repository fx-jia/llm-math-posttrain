# LLM Math Post-training V2

面向小型语言模型数学推理的可复现后训练实验。项目不再只验证 SFT、DPO、GRPO
能否运行，而是研究：在固定训练与 rollout 预算下，经过统一验证、难度筛选的后训练，
是否能稳定超过 response-only SFT。

## V2 的关键变化

- 单一数学验证器：数据筛选、DPO 标签、RLVR reward 和评测共用
  `src/math_verifier.py`，支持整数、小数、分数、百分数、科学计数法和 `\boxed{}`。
- Response-only SFT：prompt token 的 label 为 `-100`，不再把复述指令计入训练目标；超长样本被显式丢弃并统计。
- 同策略偏好数据：每题采样多个候选，从同一 SFT policy 中选择正确/错误回答，并按 token 长度匹配，减少 DPO 风格捷径。
- 一次 rollout 生成三类训练集：verified RFT 正样本、DPO hard pairs、RLVR policy-frontier prompts。
- 稳定 RLVR：默认 correctness-only reward、frontier 数据、DAPO loss、非对称 clip、关闭 reward std scaling，并屏蔽截断 completion。
- 可信评测：完整测试集、固定 seed、Wilson 95% 区间、exact McNemar 配对检验、pass@k 和 majority@k。
- 每个新训练/评测输出都会保存 run manifest，记录 git commit、参数和关键依赖版本。

## 研究流程

```text
GSM8K
  │
  ├─ response-only SFT
  │      │
  │      └─ K-way policy rollouts + shared verifier
  │             ├─ verified correct ──→ RFT
  │             ├─ correct/wrong, length matched ──→ DPO
  │             └─ 0 < pass@K < 1 ──→ GRPO / Dr.GRPO / DAPO
  │
  └─ full greedy eval + sampling eval + paired statistics
```

核心实验矩阵见 `configs/experiment_matrix_v2.yaml`，研究主张、控制变量和验收标准见
`notes/upgrade_v2.md`。

## 环境

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
```

默认模型是 `Qwen/Qwen2.5-1.5B`。可通过环境变量替换：

```bash
MODEL_NAME=/path/to/model python scripts/train_sft_lora.py --train-limit 256
```

当前稳定 RLVR recipe 需要支持 `loss_type=dapo`、`scale_rewards`、`epsilon_high`
和 `mask_truncated_completions` 的新版 TRL。脚本发现版本不支持时会直接报错，避免静默降级。

## 1. 数据准备

```bash
python scripts/prepare_gsm8k.py
python scripts/prepare_gsm8k.py --check
python scripts/inspect_gsm8k.py
```

如果已有旧版 processed 数据，可离线创建 V2 split：

```bash
python scripts/prepare_gsm8k.py --from-existing
```

准备脚本使用 seed 42 从原训练集固定留出 256 条 dev，生成
`gsm8k_train_core.jsonl`、`gsm8k_dev.jsonl` 和原始完整 train/test。所有 V2 训练默认使用
`train_core`；test 只用于最终报告。

## 2. Response-only SFT

先做烟雾训练：

```bash
python scripts/train_sft_lora.py \
  --train-limit 256 \
  --output-dir outputs/sft_v2_smoke \
  --seed 42
```

全量训练：

```bash
python scripts/train_sft_lora.py \
  --train-limit 7217 \
  --output-dir outputs/sft_v2_full \
  --max-length 768 \
  --seed 42
```

## 3. 一次 rollout 构建 RFT、DPO 与 RLVR 数据

```bash
SFT_ADAPTER_DIR=outputs/sft_v2_full python scripts/build_dpo_pairs.py \
  --input-limit 2000 \
  --num-candidates 8 \
  --max-pairs 1000 \
  --seed 42
```

输出文件：

| 文件 | 用途 |
|---|---|
| `data/processed/rft_correct_sft.jsonl` | 当前策略生成且验证正确的 RFT completion |
| `data/processed/dpo_pairs_sft.jsonl` | 同策略、长度匹配的正确/错误偏好对 |
| `data/processed/rlvr_frontier_sft.jsonl` | 组内同时出现正确和错误结果的 RLVR prompts |
| `data/processed/rollout_stats_sft.jsonl` | 每题 pass rate、格式率和长度统计 |

旧版 `dpo_pairs_sft.jsonl` 不是同策略 pair。DPO 训练默认拒绝旧格式，必须先重新构造，
或显式使用 `--allow-legacy-pairs` 做历史复现。

## 4. RFT 与 DPO 基线

RFT 从同一个 SFT adapter 继续训练：

```bash
python scripts/train_sft_lora.py \
  --train-path data/processed/rft_correct_sft.jsonl \
  --init-adapter outputs/sft_v2_full \
  --train-limit 1000 \
  --learning-rate 1e-5 \
  --output-dir outputs/rft_v2 \
  --seed 42
```

DPO：

```bash
SFT_ADAPTER_DIR=outputs/sft_v2_full python scripts/train_dpo_lora.py \
  --train-limit 1000 \
  --output-dir outputs/dpo_v2 \
  --seed 42
```

## 5. Difficulty-frontier RLVR

默认使用 DAPO 风格 token-level loss，`epsilon_high=0.28`、不按组内标准差缩放 reward，
且只使用答案正确性作为 reward。格式奖励不会再伪装成推理进步。

```bash
SFT_ADAPTER_DIR=outputs/sft_v2_full python scripts/train_grpo_lora.py \
  --train-path data/processed/rlvr_frontier_sft.jsonl \
  --train-limit 256 \
  --max-steps 100 \
  --num-generations 4 \
  --loss-type dapo \
  --output-dir outputs/rlvr_dapo_seed42 \
  --seed 42
```

关键消融只修改 `--loss-type`：

```bash
--loss-type grpo
--loss-type dr_grpo
--loss-type dapo
```

训练时重点观察 `frac_reward_zero_std`、entropy、KL、clip ratio、completion clipped ratio
和输出长度。frontier 数据只是 SFT 起点上的离线难度估计；长训练中策略能力边界会移动，后续应周期性重建该数据。

日志诊断：

```bash
python scripts/summarize_rlvr_log.py logs/rlvr_dapo/train.log
```

诊断脚本会在零方差 step 超过 10% 或截断 completion 超过 5% 时给出告警。

## 6. 评测

不传 `--limit` 时默认评测完整 1,319 条测试集：

```bash
python scripts/eval_base.py --seed 42
python scripts/eval_lora.py --adapter-dir outputs/sft_v2_full --seed 42
python scripts/eval_lora.py --adapter-dir outputs/dpo_v2 --seed 42
python scripts/eval_lora.py --adapter-dir outputs/rlvr_dapo_seed42 --seed 42
```

训练过程中应使用 dev 选择 checkpoint：

```bash
python scripts/eval_lora.py \
  --adapter-dir outputs/rlvr_dapo_seed42/checkpoint-50 \
  --data-path data/processed/gsm8k_dev.jsonl \
  --seed 42
```

小样本调试会按 seed 随机抽样，不再固定取前 N 条：

```bash
python scripts/eval_lora.py --adapter-dir outputs/sft_v2_full --limit 100 --seed 42
```

采样评测：

```bash
python scripts/eval_sampling.py \
  --adapter-dir outputs/rlvr_dapo_seed42 \
  --limit 100 \
  --num-samples 8 \
  --pass-k 1 4 8 \
  --seed 42
```

汇总与配对比较：

```bash
python scripts/summarize_eval.py \
  outputs/eval_sft_v2_full_gsm8k_test_full.jsonl \
  outputs/eval_rlvr_dapo_seed42_gsm8k_test_full.jsonl
python scripts/analyze_eval_diff.py \
  outputs/eval_sft_v2_full_gsm8k_test_full.jsonl \
  outputs/eval_rlvr_dapo_seed42_gsm8k_test_full.jsonl \
  --name-a SFT --name-b DAPO
```

`summarize_eval.py` 会输出 accuracy 的 Wilson 95% 区间和 tokens/correct；
`analyze_eval_diff.py` 会输出 exact McNemar p-value。

## V1 结果应如何理解

旧版前 100 条结果为 Base 51%、SFT 68%、DPO 69%、GRPO 62%。逐样本比较显示：

- DPO 相比 SFT 是 5 题变对、4 题变错，exact McNemar `p=1.0`，不能据此声称 DPO 有效。
- GRPO 相比 SFT 是 0 题改善、6 题退化。
- GRPO 的 20 个 step 中有 9 个 step 组内 reward 方差为零。

这些文件保留为历史结果，但 V2 的算法结论必须在新 verifier、新数据构造和完整评测上重新获得。
