# Experiment Log

## 2026-05-20 Step 1: Project Initialization

Environment:
- Platform: AutoDL
- GPU: NVIDIA GeForce RTX 5090 32GB
- Python: 3.11 in conda environment llm-posttrain
- PyTorch: CUDA available

Project objective:
Build a small but complete LLM post-training project for mathematical reasoning, including SFT, DPO, and GRPO/RLVR experiments.

Why this project:
Math reasoning tasks have verifiable final answers, so they are suitable for objective evaluation and rule-based reinforcement learning rewards.

Current status:
Project directory, Git repository, README, and experiment log initialized.

## 2026-05-20 Step 2: GSM8K Data Preprocessing

What we did:
Downloaded GSM8K through the Hugging Face mirror and converted it into a unified JSONL format for later SFT, DPO, and GRPO experiments.

What we learned:
GSM8K provides both reasoning traces and final answers, making it suitable for supervised fine-tuning and rule-based correctness rewards. AutoDL may require HF_ENDPOINT=https://hf-mirror.com to access Hugging Face resources.

## 2026-05-20 Step 3: GSM8K Data Inspection

What we did:
Inspected GSM8K processed samples, including rough sequence length, answer format, and SFT prompt consistency.

What we learned:
Before training, data statistics are needed to choose max sequence length, output format constraints, and evaluation logic.

## 2026-05-20 Step 4: Base Model Smoke Test

What we did:
Loaded Qwen2.5-1.5B with BF16 on RTX 5090 and ran one zero-shot math reasoning generation.

What we learned:
Before training, a smoke test verifies model download, tokenizer compatibility, CUDA placement, dtype, generation speed, and memory usage.

## 2026-05-20 Step 5: Base Model Evaluation

What we did:
Built an automatic evaluation script for the base model on a small GSM8K test subset.

What we learned:
A baseline is necessary before fine-tuning, otherwise later SFT, DPO, and GRPO results cannot be interpreted as actual improvements.

## 2026-05-20 Step 6: LoRA SFT Smoke Training

What we did:
Ran a small LoRA supervised fine-tuning smoke test on 256 GSM8K training samples.

What we learned:
A small training run validates tokenization, LoRA adapter injection, loss logging, checkpoint saving, and GPU memory usage before launching larger experiments.

## 2026-05-20 Step 7: SFT Smoke Adapter Evaluation

What we did:
Loaded the smoke LoRA adapter and evaluated it on the same 20 GSM8K test examples as the base model.

What we learned:
Adapter evaluation checks whether the saved LoRA weights can be reloaded and compared under the same prompt, decoding, and metrics as the baseline.

## 2026-05-20 Step 8: Full LoRA SFT Training

What we did:
Trained a LoRA rank-8 SFT adapter on the full GSM8K training split with max_length=512.

What we learned:
The full SFT run is the first real post-training checkpoint. It will be compared against the base model and the smoke adapter under the same evaluation protocol.


## 2026-05-21 Step 9: Full SFT Evaluation

Results on the first 100 GSM8K test examples:
- Base exact_match: 0.5000
- SFT LoRA r=8 exact_match: 0.6700
- Base format_rate: 0.8800
- SFT LoRA r=8 format_rate: 0.9700
- Base avg_output_tokens: 181.7
- SFT LoRA r=8 avg_output_tokens: 129.1

What we learned:
LoRA SFT substantially improved answer accuracy and output format compliance on the sampled GSM8K evaluation set. It also made outputs more concise, which is useful for automatic answer extraction and later preference/RL training.

## 2026-05-21 Step 10: DPO Preference Pair Construction

What we did:
Sampled responses from the SFT adapter and constructed DPO preference pairs using GSM8K ground-truth solutions as chosen responses and incorrect model generations as rejected responses.

What we learned:
For math reasoning, preference data can be bootstrapped with verifiable final answers. This is not human RLHF; it is automatically constructed preference optimization data.

## 2026-05-21 Step 11: DPO Smoke Training

What we did:
Ran a small DPO training job on 200 automatically constructed preference pairs, starting from the SFT LoRA adapter.

What we learned:
DPO uses chosen/rejected pairs to optimize the policy against a frozen reference model. For this project, the reference model is the SFT checkpoint before DPO.

## 2026-05-21 Step 12: DPO Smoke Evaluation

What we did:
Evaluated the DPO smoke adapter on the same 100 GSM8K test examples used for Base and SFT evaluation.

What we learned:
DPO evaluation must use the same prompt, decoding method, and answer extraction logic as Base/SFT. Small DPO runs may not improve exact match, but they reveal whether preference optimization preserves or damages the SFT model's behavior.

## 2026-05-21 Step 13: GRPO Reward Function Test

What we did:
Prepared a GRPO training subset and implemented rule-based rewards for answer correctness, output format, and excessive length.

What we learned:
For math RLVR, reward design is the central part. Correctness reward provides the main learning signal, while format and length rewards stabilize automatic evaluation and discourage unusable generations.

## 2026-05-21 Step 14: GRPO Smoke Training

What we did:
Ran a small GRPO/RLVR smoke training job from the SFT LoRA adapter using rule-based correctness, format, and length rewards.

What we learned:
GRPO samples multiple completions per prompt and uses relative rewards within the group to update the policy. Some batches have zero reward variance, which provides little learning signal, while batches with mixed correct and incorrect generations provide useful RLVR updates.

## 2026-05-21 Step 15: GRPO Smoke Evaluation

What we did:
Evaluated the GRPO smoke adapter on the same 100 GSM8K test examples and compared it with Base, SFT, and DPO.

What we learned:
RLVR/GRPO must be evaluated under the same deterministic protocol as SFT and DPO. A small GRPO run is mainly used to verify whether reward-based training preserves or improves the SFT checkpoint.



## 2026-05-21 Step 16: GRPO Error Analysis

What we did:
Compared SFT and GRPO predictions on the same 100 GSM8K evaluation examples and identified regressions and improvements.

What we learned:
A post-training method should not be judged only by aggregate accuracy. Case-level analysis reveals whether changes come from better reasoning, worse arithmetic, format drift, or answer extraction issues.

## 2026-05-21 Step 17: Numeric Answer Normalization Fix

What we did:
Added a stricter numeric normalization utility and recomputed evaluation metrics from saved prediction files.

What we learned:
String-level exact match can underestimate math accuracy when predictions such as 26.00 and 26 are numerically equivalent. Evaluation bugs must be fixed before drawing conclusions from post-training comparisons.

## 2026-05-21 Step 18: Evaluation Script Fix

What we did:
Updated Base and LoRA evaluation scripts to use the shared numeric answer normalization utility.

What we learned:
Evaluation logic should be centralized. Otherwise, different scripts may silently use inconsistent correctness criteria and produce misleading comparisons.


## 2026-05-21 Step 19: Re-evaluation with Fixed Numeric Normalization

What we did:
Regenerated the 100-example evaluation files for Base, SFT, DPO, and GRPO using the corrected numeric answer normalization logic.

Results:
- Base exact_match: 0.5100, format_rate: 0.8800
- SFT exact_match: 0.6800, format_rate: 0.9700
- DPO smoke exact_match: 0.6900, format_rate: 0.9900
- GRPO smoke exact_match: 0.6200, format_rate: 0.9800

What we learned:
SFT provides the main accuracy gain, DPO slightly improves the SFT checkpoint on this sample, while the short GRPO smoke run validates the RLVR pipeline but hurts accuracy under the current reward/training setup.


## 2026-05-21 Step 20: Report V1

What we did:
Created the first project report summarizing data, methods, evaluation results, error analysis, limitations, and next steps.

What we learned:
A credible LLM project needs not only checkpoints, but also reproducible records, fair comparisons, error analysis, and honest limitations.
