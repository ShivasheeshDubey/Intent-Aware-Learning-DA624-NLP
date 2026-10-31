#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import spacy
from datasets import load_dataset
from tqdm import tqdm

from intent_aware.content_words import extract_content_words, keep_reasonable_texts
from intent_aware.io_utils import write_jsonl
from intent_aware.model_utils import load_tokenizer


def load_stream(name: str, seed: int):
    if name == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    elif name == "openwebtext":
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    else:
        raise ValueError(name)
    return ds.shuffle(seed=seed, buffer_size=10_000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["c4", "openwebtext"], default="c4")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--max-samples", type=int, default=120)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--min-chars", type=int, default=50)
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    tokenizer = load_tokenizer(args.model)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    ds = load_stream(args.dataset, args.seed)

    rows = []
    texts = keep_reasonable_texts(
        ds,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_samples=args.max_samples * 3,
    )
    for text in tqdm(texts, desc=f"Extracting {args.dataset} content words"):
        targets = extract_content_words(text, nlp, tokenizer, args.max_length)
        if not targets:
            continue
        rows.append({"text": text, "targets": targets, "source": args.dataset})
        if len(rows) >= args.max_samples:
            break

    write_jsonl(args.output, rows)
    n_targets = sum(len(x["targets"]) for x in rows)
    print(f"Wrote {len(rows)} sequences to {args.output}")
    print(f"Valid content-word targets: {n_targets}")
    print(f"Mean targets/sequence: {n_targets / max(len(rows), 1):.2f}")


if __name__ == "__main__":
    main()
