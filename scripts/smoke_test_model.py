import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")


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

    prompt = (
        "You are a helpful math reasoning assistant.\n"
        "Solve the following problem step by step, and put the final answer after 'Final Answer:'.\n\n"
        "Problem:\n"
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?\n\n"
        "Solution:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

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
