#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from tqdm import tqdm

from intent_aware.concept_sets import topk_word_candidates
from intent_aware.io_utils import read_jsonl, write_jsonl
from intent_aware.model_utils import load_quantized_causal_lm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--max-candidates", type=int, default=30)
    p.add_argument("--max-targets-per-sequence", type=int, default=4)
    args = p.parse_args()

    tokenizer, model = load_quantized_causal_lm(args.model)
    rows_out = []

    for row in tqdm(read_jsonl(args.input), desc="Generating top-k candidates"):
        enc = tokenizer(
            row["text"],
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[0]

        new_targets = []
        for t in row.get("targets", [])[: args.max_targets_per_sequence]:
            pos = int(t["position"])
            if not (0 < pos < enc["input_ids"].shape[1]):
                continue
            actual_id = int(enc["input_ids"][0, pos].item())
            if actual_id != int(t["token_id"]):
                continue
            probs_at_pos = torch.softmax(logits[pos - 1].float(), dim=-1)
            candidates = topk_word_candidates(
                logits[pos - 1],
                tokenizer,
                top_k=args.top_k,
                max_candidates=args.max_candidates,
            )
            target = dict(t)
            target["target_base_prob"] = float(probs_at_pos[actual_id].item())
            target["candidates"] = candidates
            new_targets.append(target)

        if new_targets:
            rows_out.append({**row, "targets": new_targets})

    write_jsonl(args.output, rows_out)
    print(f"Wrote {len(rows_out)} sequences to {args.output}")
    print(f"Targets with candidate pools: {sum(len(x['targets']) for x in rows_out)}")


if __name__ == "__main__":
    main()
