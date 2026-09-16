# LLM Math Post-training V3

这个项目研究一个更清晰、也更适合面试讲述的问题：在单张 32GB 级 GPU 的有限算力下，
能否通过 SFT → verified RFT/DPO → RLVR，把较新的 4B Base 数学模型从 **xx** 提升到
**xx**，并在困难、近期的 benchmark 上达到或超过同代 9B 模型。

- 本地可训练模型：`Qwen/Qwen3.5-4B-Base`
- 同尺寸官方控制组：`Qwen/Qwen3.5-4B`
- 主要大模型基线：`Qwen/Qwen3.5-9B`，通过兼容 OpenAI 的 API 推理
- 可选能力上限：`Qwen/Qwen3.5-27B`，仅 API 推理
- headline benchmark：HMMT February 2026
- 次级 benchmark：AIME 2026、Apex 2025
- 稳定性 benchmark：MATH-500

完整主张、风险与验收标准见 [V3 研究计划](notes/upgrade_v3.md)，固定实验配置见
[`experiment_matrix_v3.yaml`](configs/experiment_matrix_v3.yaml)。仓库不会预填尚未实际跑出的成绩。

## 服务器同步与环境

每个逻辑修改都会形成独立 Git commit。服务器端使用：

```bash
git pull --ff-only
python -m pip install -r requirements.txt
python -m unittest discover -v
```

Qwen3.5 的原生支持需要 `transformers>=5.17`。训练使用 BF16、LoRA 和 gradient
checkpointing；如果环境支持 `causal-conv1d`/FLA 加速内核，可以额外安装以提高混合线性注意力吞吐，
但 V3 代码不把非通用内核作为硬依赖。

API key 只通过环境变量提供，禁止写入仓库：

```bash
cp .env.example .env
export TOGETHER_API_KEY='your-key'
export API_BASE_URL='https://api.together.xyz/v1'
export API_MODEL_NAME='Qwen/Qwen3.5-9B'
# 正式实验建议把 main 换成记录在 manifest 中的模型 commit SHA
export MODEL_REVISION='main'
```

`.env`、数据、checkpoint、输出和日志均已被 Git 忽略。

## 1. 准备并冻结数据

```bash
python scripts/prepare_v3_train_data.py \
  --sft-limit 40000 \
  --min-correctness 2 \
  --seed 42

python scripts/prepare_v3_benchmarks.py
```

训练来源分别是 `open-r1/OpenR1-Math-220k` 和去重处理后的
`open-r1/DAPO-Math-17k-Processed`。评测准备脚本会生成独立 benchmark 文件以及合并的
`matharena_v3.jsonl`。每个数据文件旁都有包含 revision、参数和 Git commit 的 manifest。

训练前分别过滤 SFT 数据和 RLVR prompt：

```bash
python scripts/check_decontamination.py \
  --train-path data/processed/openr1_sft_v3.jsonl \
  --report data/processed/decontamination_openr1_v3.json \
  --filtered-output data/processed/openr1_sft_v3_clean.jsonl

python scripts/check_decontamination.py \
  --train-path data/processed/dapo_math_v3.jsonl \
  --report data/processed/decontamination_dapo_v3.json \
  --filtered-output data/processed/dapo_math_v3_clean.jsonl
```

词面 Jaccard 阈值默认是 0.8。报告需要与实验结果一起保存；如命中较多，应人工复核而不是只看总数。

## 2. 先冻结三个起点基线

所有采样评测默认使用 4 samples、temperature 1.0、top-p 0.95、top-k 20、
max new tokens 4096。不要只比较不同模型各自最有利的 decoding。

```bash
# 4B Base
python scripts/eval_sampling.py \
  --data-path data/processed/hmmt_feb_2026.jsonl \
  --num-samples 4 --pass-k 1 4 --seed 42

# 官方同尺寸 4B post-trained control
MODEL_NAME=Qwen/Qwen3.5-4B python scripts/eval_sampling.py \
  --data-path data/processed/hmmt_feb_2026.jsonl \
  --num-samples 4 --pass-k 1 4 --seed 42

# 9B API baseline
python scripts/eval_api.py \
  --data-path data/processed/hmmt_feb_2026.jsonl \
  --num-samples 4 --seed 42
```

如 API 平台提供 27B，再显式替换 `--model Qwen/Qwen3.5-27B`。API 脚本保存每个原始回答、
reasoning（若供应商返回）、token usage、延迟、错误以及不含密钥的 manifest。

## 3. Response-only SFT

先用小样本验证显存和依赖：

```bash
python scripts/train_sft_lora.py \
  --train-path data/processed/openr1_sft_v3_clean.jsonl \
  --train-limit 128 \
  --lora-rank 32 --lora-alpha 64 \
  --output-dir outputs/sft_v3_smoke
```

正式训练：

```bash
python scripts/train_sft_lora.py \
  --train-path data/processed/openr1_sft_v3_clean.jsonl \
  --train-limit 40000 \
  --max-length 4096 \
  --lora-rank 32 --lora-alpha 64 \
  --output-dir outputs/sft_v3_seed42 \
  --seed 42
```

prompt token 全部用 `-100` 屏蔽。Qwen3.5 是混合注意力架构，LoRA 除传统
q/k/v/o 和 MLP 外，也覆盖 DeltaNet 的 `in_proj_*` 与 `out_proj`。

## 4. 同策略 rollout：一次生成三类数据

```bash
SFT_ADAPTER_DIR=outputs/sft_v3_seed42 python scripts/build_dpo_pairs.py \
  --train-path data/processed/dapo_math_v3_clean.jsonl \
  --input-limit 4000 \
  --num-candidates 8 \
  --max-pairs 2000 \
  --seed 42
```

输出包括：

| 文件 | 用途 |
|---|---|
| `rft_correct_sft.jsonl` | 当前策略生成且 verifier 判对的 RFT completion |
| `dpo_pairs_sft.jsonl` | 同策略、长度匹配的正确/错误 pair |
| `rlvr_frontier_sft.jsonl` | 同一组内既有对也有错的 policy-frontier prompt |
| `rollout_stats_sft.jsonl` | 每题 pass rate、格式率和 token 长度 |

## 5. RFT、DPO 与 RLVR

三个方法都从同一 SFT adapter 起步：

```bash
# RFT
python scripts/train_sft_lora.py \
  --train-path data/processed/rft_correct_sft.jsonl \
  --init-adapter outputs/sft_v3_seed42 \
  --train-limit 2000 --learning-rate 1e-5 \
  --output-dir outputs/rft_v3_seed42 --seed 42

# DPO
SFT_ADAPTER_DIR=outputs/sft_v3_seed42 python scripts/train_dpo_lora.py \
  --train-limit 2000 \
  --output-dir outputs/dpo_v3_seed42 --seed 42

# DAPO-style RLVR
SFT_ADAPTER_DIR=outputs/sft_v3_seed42 python scripts/train_grpo_lora.py \
  --train-path data/processed/rlvr_frontier_sft.jsonl \
  --train-limit 512 --max-steps 200 --num-generations 4 \
  --loss-type dapo \
  --output-dir outputs/rlvr_dapo_v3_seed42 --seed 42
```

方法结论需要重复 seed 42、43、44。`loss_type=grpo` 是最关键的 RL 消融；RFT 如果与
RLVR 相当，应把结论归因到 verified sampling/filtering，而不是格式奖励或 RL 标签。

## 6. 固定协议最终评测

```bash
python scripts/eval_sampling.py \
  --adapter-dir outputs/rlvr_dapo_v3_seed42 \
  --data-path data/processed/hmmt_feb_2026.jsonl \
  --num-samples 4 --pass-k 1 4 --seed 42

python scripts/eval_sampling.py \
  --adapter-dir outputs/rlvr_dapo_v3_seed42 \
  --data-path data/processed/aime_2026.jsonl \
  --num-samples 4 --pass-k 1 4 --seed 42

python scripts/eval_sampling.py \
  --adapter-dir outputs/rlvr_dapo_v3_seed42 \
  --data-path data/processed/math_500.jsonl \
  --num-samples 4 --pass-k 1 4 --seed 42
```

评测、rollout label 和 RL reward 共用 `src/math_verifier.py`。它先做精确数值比较，再用
Math-Verify 处理根式、LaTeX 与符号等价，避免 HMMT 的符号答案被“最后一个数字”误判。

汇总与逐题配对分析仍可使用：

```bash
python scripts/summarize_eval.py outputs/<result>.jsonl
python scripts/analyze_eval_diff.py outputs/<a>.jsonl outputs/<b>.jsonl \
  --name-a SFT --name-b DAPO
```

最终项目表述应是：“4B Base 为 **xx**，经过 SFT/RFT/RLVR 后达到 **xx**；同协议下官方
4B 为 **xx**、API 9B 为 **xx**。”在这些输出真实存在之前，保留 `xx`。
