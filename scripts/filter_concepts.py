#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from tqdm import tqdm

from intent_aware.concept_sets import build_filter_prompt, parse_allowed_list
from intent_aware.io_utils import read_jsonl, write_jsonl
from intent_aware.model_utils import load_quantized_causal_lm


def make_chat_prompt(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-synonyms", type=int, default=10)
    args = p.parse_args()

    tokenizer, model = load_quantized_causal_lm(args.model)
    tokenizer.padding_side = "left"

    rows = list(read_jsonl(args.input))
    jobs = []
    for ri, row in enumerate(rows):
        for ti, target in enumerate(row.get("targets", [])):
            target_word = target["word"]
            allowed = [
                c["word"] for c in target.get("candidates", [])
                if c["word"].lower() != target_word.lower()
            ]
            prompt = build_filter_prompt(row["text"], target_word, allowed)
            jobs.append((ri, ti, allowed, make_chat_prompt(tokenizer, prompt)))

    selected = {}
    for start in tqdm(range(0, len(jobs), args.batch_size), desc="Filtering contextual synonyms"):
        chunk = jobs[start:start + args.batch_size]
        prompts = [x[3] for x in chunk]
        enc = tokenizer(prompts, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        tail = generated[:, enc["input_ids"].shape[1]:]
        responses = tokenizer.batch_decode(tail, skip_special_tokens=True)
        for (ri, ti, allowed, _), response in zip(chunk, responses):
            selected[(ri, ti)] = parse_allowed_list(response, allowed)[: args.max_synonyms]

    total_targets = 0
    multi_member = 0
    for ri, row in enumerate(rows):
        cleaned = []
        for ti, target in enumerate(row.get("targets", [])):
            synonyms = selected.get((ri, ti), [])
            candidate_map = {c["word"].lower(): c for c in target.get("candidates", [])}
            concept_ids = [int(target["token_id"])]
            prior_probs = [float(target.get("target_base_prob", 1e-12))]
            kept_synonyms = []
            for word in synonyms:
                cand = candidate_map.get(word.lower())
                token_id = int(cand["token_id"]) if cand is not None else None
                if token_id is not None and token_id not in concept_ids:
                    concept_ids.append(token_id)
                    prior_probs.append(float(cand.get("prob", 1e-12)))
                    kept_synonyms.append(word)
            z = sum(prior_probs) or 1.0
            prior_probs = [p / z for p in prior_probs]
            out = {k: v for k, v in target.items() if k not in {"candidates", "target_base_prob"}}
            out["synonyms"] = kept_synonyms
            out["concept_token_ids"] = concept_ids
            out["concept_prior"] = prior_probs
            cleaned.append(out)
            total_targets += 1
            if len(concept_ids) > 1:
                multi_member += 1
        row["targets"] = cleaned

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} sequences to {args.output}")
    print(f"Concept targets: {total_targets}")
    print(f"Targets with >=1 contextual synonym: {multi_member} ({100*multi_member/max(total_targets,1):.1f}%)")


if __name__ == "__main__":
    main()
