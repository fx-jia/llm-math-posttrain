# Repository instructions

## Project objective

This repository studies whether compute-efficient post-training can make
`Qwen/Qwen3.5-4B-Base` match or exceed a newer, larger `Qwen/Qwen3.5-9B`
inference-only baseline on hard, recent mathematics benchmarks. The primary
headline benchmark is HMMT February 2026. AIME 2026 and Apex 2025 are
secondary stress tests; MATH-500 is a regression/stability check.

## Scientific rules

- Never invent, interpolate, or backfill a score. Use `xx` or mark a result as
  pending until an output artifact exists.
- Do not train, select checkpoints, tune prompts, or tune decoding on final
  evaluation answers. Run the decontamination audit before training.
- Use `src/math_verifier.py` for data labeling, rewards, and evaluation. Do not
  add a stage-specific correctness parser.
- Compare models with the same prompt semantics, sampling count, temperature,
  top-p, top-k, and token limit. Record any unavoidable provider difference.
- Report Base, response-only SFT, verified RFT, length-matched DPO, and RLVR.
  Keep `Qwen/Qwen3.5-4B` as the same-size official post-trained control.
- Use seeds 42, 43, and 44 for claims about training methods. Preserve raw
  generations, manifests, package versions, dataset revisions, and input
  hashes.
- Treat public leaderboard scores only as preregistered reference targets.
  Project claims must use this repository's reruns under the fixed protocol.

## Engineering rules

- Keep model, dataset, and provider defaults in shared modules/configs rather
  than duplicating strings.
- Add dependency-free unit tests for normalization, verification, filtering,
  or statistics logic. Avoid downloading models or datasets in unit tests.
- Before a commit run:

  ```bash
  python3 -m unittest discover -v
  python3 -m compileall -q src scripts tests
  git diff --check
  ```

- Make one Git commit per logical change with an imperative message. Do not
  commit datasets, checkpoints, generated outputs, logs, `.env`, or API keys.
- Preserve unrelated user changes. Do not rewrite history or force-push.
- Server workflows should start with `git pull --ff-only` and use environment
  variables for local model paths and credentials.
