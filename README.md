# LLM Math Post-training

This project studies post-training methods for mathematical reasoning on small language models.

Core goal:
- Build a reproducible post-training pipeline for math reasoning.
- Compare Base, SFT, DPO, and GRPO/RLVR.
- Use answer correctness and output format as measurable evaluation signals.

Planned model:
- Qwen or DeepSeek distilled Qwen small model.

Planned methods:
- LoRA / QLoRA supervised fine-tuning
- Preference optimization with DPO
- Reinforcement learning with verifiable rewards using GRPO

Main metrics:
- Exact Match of final answer
- Format compliance rate
- Average output length
- Inference latency
- GPU memory usage
