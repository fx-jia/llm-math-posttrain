# V3 研究计划：有限算力上的小模型数学后训练

## 一句话主张

在单张 32GB 级 GPU 上，对较新的 `Qwen/Qwen3.5-4B-Base` 依次进行
response-only SFT、同策略 verified RFT/DPO 与 difficulty-frontier RLVR，研究它在
HMMT February 2026 上能否从基座的 **xx** 提升至 **xx**，并达到或超过只通过 API
推理的 `Qwen/Qwen3.5-9B` 在同一评测协议下的成绩。

这里的 `xx` 必须由本仓库产生的原始评测文件填写，不能用排行榜数字替代。

## 为什么这样选

- **4B Base 可训练**：参数量足够小，LoRA + BF16 + gradient checkpointing 可在有限算力上执行；Base 版本也让后训练增益更容易归因。
- **4B Instruct 是必要控制组**：它回答“自己做的后训练是否超过官方同尺寸后训练”，避免只与弱 Base 比较。
- **9B 是主要目标**：同代、同家族、参数更大，架构与 tokenizer 差异较少；只做 API 推理，不占本地训练显存。
- **27B 只作为能力上限**：它不是成功判据，且仅在 API 预算允许时运行。
- **HMMT 2026 是 headline**：比 GSM8K 更难且更新；AIME 2026、Apex 2025 检查迁移，MATH-500 检查是否遗忘稳定能力。

公开 MathArena 页面曾给出 Qwen3.5-9B 在 HMMT February 2026 上约 71.21%、
Qwen3.5-27B 约 81.06% 的结果。这些数字仅用于预注册目标和预算判断；供应商、prompt、
采样配置和 verifier 不同都可能改变结果，因此最终比较必须重跑。

## 实验阶段

1. 冻结 benchmark 与数据 revision，生成 manifest。
2. 对 OpenR1 SFT 数据和 DAPO prompt 分别做评测集 overlap 审计并过滤。
3. 评测 4B Base、官方 4B、API 9B；三者共享 prompt 语义与采样参数。
4. 训练 4B Base 的 response-only LoRA SFT。
5. 用 SFT policy 对去污染后的 DAPO prompt 做 K-way rollout，一次产生：
   verified RFT、长度匹配 DPO pairs、policy-frontier RLVR prompts。
6. 在同一个 SFT 起点上分别训练 RFT、DPO、GRPO/DAPO。
7. 在固定 benchmark 上评测，并报告逐题原始输出、accuracy、pass@4、majority@4、
   token 开销和置信区间。

## 归因与消融

- Base → SFT：高质量监督数据的贡献。
- SFT → RFT：采样与 verifier 筛选的贡献。
- SFT → DPO：同策略偏好优化的贡献。
- SFT → GRPO/DAPO：在线可验证奖励的贡献。
- DAPO vs vanilla GRPO：loss 与稳定化配置的贡献。
- 4B 自研最佳模型 vs 官方 4B：训练方案价值。
- 4B 自研最佳模型 vs API 9B：核心参数效率结论。

所有算法结论至少跑 seed 42/43/44，并优先报告逐题配对差异。若 RFT 与 RLVR 相当，
应如实得出“收益主要来自生成与筛选”而不是强行声称 RL 有效。

## 成功标准

- 最低标准：相对 4B Base 在 HMMT 2026 上有可复现提升，且 MATH-500 不明显退化。
- 项目主张：最佳 4B 后训练模型在固定协议下达到或超过 API 9B。
- 强主张：上述结论至少在一个次级 benchmark 上也成立，并跨至少两个 seed 重复。

## 风险

- HMMT 只有 33 题，单题约 3 个百分点，必须给区间并配合多个 benchmark。
- 公共数学训练集可能包含近似题；词面去污染只是最低标准，结果中要披露阈值与命中项。
- API 服务端实现可能变化；保存模型名、日期、参数、usage 和原始返回。
- Qwen3.5 包含线性注意力模块；LoRA 必须覆盖 `in_proj_*`/`out_proj`，不能只配传统 q/k/v/o。
- 长推理会显著增加 rollout 成本；比较时必须固定 max tokens 并报告 tokens/correct。

## 主要资料

- [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- [DAPO-Math-17k-Processed](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed)
- [HMMT February 2026](https://huggingface.co/datasets/MathArena/hmmt_feb_2026)
- [MathArena](https://matharena.ai/)
