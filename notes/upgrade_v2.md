# V2 research claim and acceptance criteria

## Claim

Under a matched training-token and rollout-token budget, difficulty-frontier RLVR should improve
the SFT policy more reliably than training on unfiltered prompts, without obtaining the gain from
answer-parser drift, format-only reward, or uncontrolled response length.

The null result is useful: if verified RFT matches or beats DPO/RLVR, the dominant contribution is
candidate generation and filtering rather than the preference/RL objective.

## Required baselines

1. Base model.
2. Response-only SFT.
3. Verified rejection-sampling fine-tuning (RFT).
4. Same-policy, length-matched DPO.
5. Vanilla GRPO on the same policy-frontier prompts.
6. Dr.GRPO and DAPO-style loss ablations on those prompts.

## Controls

- Use the same `src.math_verifier` implementation for rollout labeling, rewards, and evaluation.
- Match trainable parameters, optimizer tokens, rollout tokens, prompt budget, and tuning effort.
- Run seeds 42, 43, and 44 for any result used as an algorithmic claim.
- Select checkpoints on a held-out development split, never on GSM8K test.
- Record the git commit, package versions, model revision, seed, and full arguments.

## Acceptance criteria

- Report the full 1,319-example GSM8K test split with a 95% interval.
- Use exact McNemar tests for paired greedy comparisons.
- Report pass@1, pass@4/pass@8, majority@k, average tokens, and tokens per correct answer.
- Keep truncated-completion ratio below 5% or explain and ablate it.
- Keep zero-reward-variance groups below 10% of optimizer steps when frontier filtering is enabled.
- A claimed improvement must repeat across at least two seeds and must not depend on parser changes.
