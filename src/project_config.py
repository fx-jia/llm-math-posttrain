"""Dependency-free V3 defaults shared by training and evaluation scripts."""

from __future__ import annotations


DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
DEFAULT_SAME_SIZE_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_API_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_CEILING_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_API_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL_REVISION = "main"

LEGACY_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Qwen3.5 interleaves full attention with Gated DeltaNet linear-attention
# blocks.  Adapting only q/k/v/o would leave most attention blocks frozen.
QWEN35_LORA_TARGET_MODULES = LEGACY_LORA_TARGET_MODULES + (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


def lora_target_modules(model_name: str) -> list[str]:
    """Return projection suffixes appropriate for a model family."""
    normalized = (
        model_name.casefold().replace("_", "").replace("-", "").replace(".", "")
    )
    targets = (
        QWEN35_LORA_TARGET_MODULES
        if "qwen35" in normalized
        else LEGACY_LORA_TARGET_MODULES
    )
    return list(targets)
