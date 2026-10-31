#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset

from intent_aware.model_utils import load_model_with_adapter
from intent_aware.semantic_eval import evaluate_pairs, load_men, load_simlex, load_wordsim353


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-dir", default="data/benchmarks")
    p.add_argument("--output", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--adapter", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-pairs", type=int, default=None)
    args = p.parse_args()

    tokenizer, model = load_model_with_adapter(args.base_model, args.adapter)
    root = Path(args.benchmark_dir)
    benchmarks = {
        "MEN": load_men(root / "men.txt"),
        "WordSim353": load_wordsim353(root / "wordsim353.csv"),
        "SimLex999": load_simlex(root / "simlex999.txt"),
    }
    stsb = load_dataset("sentence-transformers/stsb", split="test")
    benchmarks["STS-B"] = [(x["sentence1"], x["sentence2"], float(x["score"])) for x in stsb]

    results = {}
    for name, pairs in benchmarks.items():
        if args.max_pairs:
            pairs = pairs[: args.max_pairs]
        print(f"Evaluating {name} on {len(pairs)} pairs...")
        results[name] = evaluate_pairs(model, tokenizer, pairs, batch_size=args.batch_size)
        print(name, results[name])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
