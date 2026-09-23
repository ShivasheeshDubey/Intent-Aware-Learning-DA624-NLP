# Intent-Aware Learning for Language Models

DA624 — Natural Language Processing with Large Language Models  
IIT Guwahati

## Project

This project investigates concept-aware / intent-aware training objectives
for autoregressive language models, inspired by:

Christine Zhang, Dan Jurafsky, and Chen Shani,
"Learning Concepts, Not Tokens: Self-Supervised Semantic Alignment
for Language Models."

Standard next-token prediction rewards only the observed next token,
even when several semantically interchangeable continuations may be valid.

We study whether distributing supervision across contextually equivalent
continuations can improve semantic representations while retaining
language-modeling performance.

## Planned Pipeline

1. Extract single-token content words from natural-language sequences.
2. Construct contextual concept sets containing interchangeable continuations.
3. Fine-tune a causal language model using QLoRA.
4. Compare standard NTP against concept-aware objectives.
5. Evaluate:
   - global perplexity,
   - content-word perplexity and accuracy,
   - semantic similarity benchmarks,
   - out-of-domain generalization.
6. Investigate extensions to the original concept-training objective.

## Repository Structure

- `src/intent_aware/` — reusable project code
- `scripts/` — training and evaluation entry points
- `configs/` — experiment configurations
- `notebooks/` — exploratory notebooks
- `results/` — experiment summaries and figures
- `references/` — project references and notes

## Status

Project setup and baseline reproduction in progress.
