import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.project_config import DEFAULT_BASE_MODEL
from src.prompts import build_math_prompt, render_prompt_for_model


MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_BASE_MODEL)


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"tokenizer vocab size: {len(tokenizer)}")
    print(f"model device: {next(model.parameters()).device}")
    print(f"model dtype: {next(model.parameters()).dtype}")

    prompt = build_math_prompt(
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?"
    )
    prompt = render_prompt_for_model(tokenizer, prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    print("=" * 80)
    print("Prompt:")
    print(prompt)
    print("=" * 80)
    print("Model output:")
    print(generated)
    print("=" * 80)
    print(f"elapsed seconds: {elapsed:.2f}")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak cuda memory allocated: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
