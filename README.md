# Intent-Aware Training Objectives for Reducing Surface-Form Bias in Next-Token Prediction

DA624 — Natural Language Processing with Large Language Models, IIT Guwahati

**Team:** Harsh Vardhan Singh Bhati, Pranav Kumar Rachit, Sahastransh Mandhana, Shivasheesh Dubey

## What this project does

Standard next-token prediction rewards only the observed next token even when several continuations are contextually interchangeable. This project implements a reduced-scale concept-aware training pipeline inspired by Zhang, Jurafsky, and Shani (2026):

1. Stream C4 / OpenWebText.
2. Extract single-token NOUN/VERB/ADJ targets with spaCy + Qwen tokenizer alignment.
3. Use `Qwen/Qwen3-1.7B-Base` to obtain top-k next-token candidates at each target position.
4. Use `Qwen/Qwen3-1.7B` to filter candidates into contextual synonym/concept sets.
5. Fine-tune the base model with 4-bit NF4 QLoRA under standard NTP or concept-aware objectives.
6. Evaluate global PPL, content-word PPL/accuracy, concept mass, OOD OpenWebText behavior, semantic-similarity Spearman correlation, and paired-bootstrap CIs.

The repository also implements an optional **context-preservation KL regularizer**. It stores the base model's relative distribution within each concept set and discourages fine-tuning from unnecessarily destroying those contextual preferences. Set `--preserve-beta 0` for the Zhang-style objective.

## Core objective

For a contextual concept set `S_t`:

`P(S_t | x_<t) = sum_{w in S_t} p(w | x_<t)`

The concept loss is `-log P(S_t | x_<t)` while set mass is below 0.6, and 0 once the threshold is reached. Training uses:

`L = (1 - lambda) * L_NTP + lambda * L_concept + beta * KL(q || r)`

where the KL term is optional.

## Repository layout

```text
configs/                 emergency and paper-scale experiment settings
notebooks/               Colab submission pipeline
references/              source/reproduction notes
results/                 generated metrics + development smoke test
scripts/
  prepare_data.py         C4/OpenWebText content-word extraction
  generate_candidates.py base-model top-k continuation pools
  filter_concepts.py      instruction-model contextual synonym filtering
  split_data.py           deterministic train/val/test split
  train_model.py          QLoRA NTP/concept/extension training
  evaluate_model.py       global + content-word evaluation
  evaluate_similarity.py MEN/WordSim353/SimLex999/STS-B
  compare_results.py      paired-bootstrap intervals
  render_report.py        generate SUBMISSION_REPORT.md from results
src/intent_aware/         reusable project library
```

## Fast Colab run for tonight

Use `notebooks/01_submission_pipeline.ipynb`. It is designed for a Tesla T4 and deliberately uses a small experiment:

- 120 C4 sequences
- 60 OpenWebText sequences
- max sequence length 256
- at most 4 concept targets per C4 sequence
- top-100 LM candidates, keep up to 30 whole-word candidates
- up to 10 filtered synonyms
- NTP baseline (`lambda=0`)
- concept model (`lambda=0.5`)
- context-preserving extension (`lambda=0.5`, `beta=0.1`)
- 2 epochs, seed 42

For a larger run, use `configs/full.json` as the target specification.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -e .
```

Qwen3 requires a recent Transformers release; this project pins `transformers>=4.51.0`.

## One-command Linux/Colab execution

```bash
bash scripts/run_fast_pipeline.sh
```

The script writes evaluation files into `results/` and then generates `SUBMISSION_REPORT.md`.

## Development sanity check already obtained

For `This website is safe to`, Qwen3-1.7B-Base assigned probability `0.274158` to `browse` and total probability mass `0.649606` to the manual concept set `{browse, search, visit, surf, navigate, access, view, click}`. Exact-token NLL was `1.2941`; raw concept NLL was `0.4314`; because concept mass exceeded the `0.6` threshold, thresholded concept loss was `0.0`.

This is only a sanity check, not a benchmark result.

## Important interpretation note

The project is a **reduced-scale reproduction and extension**, not a claim to reproduce the original paper's absolute results. The emergency configuration is intentionally small because it is intended to finish on free Colab/Kaggle hardware. Use the supplied full config and multiple seeds for stronger conclusions.

## Primary reference

Christine Zhang, Dan Jurafsky, Chen Shani. *Learning Concepts, Not Tokens: Self-Supervised Semantic Alignment for Language Models.* arXiv:2603.29123, 2026.

Official implementation used as a methodological reference: https://github.com/christine-zhang1/learning-concepts
