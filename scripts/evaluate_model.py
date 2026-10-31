#!/usr/bin/env python
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_aware.evaluation import EvalDataset, evaluate_model
from intent_aware.model_utils import load_model_with_adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--adapter", default=None)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()

    tokenizer, model = load_model_with_adapter(args.base_model, args.adapter)
    ds = EvalDataset(args.data, tokenizer, args.max_length, args.max_samples)
    metrics = evaluate_model(model, tokenizer, ds, batch_size=args.batch_size)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")

    for key in ["global_perplexity", "content_perplexity", "content_accuracy", "mean_concept_mass"]:
        print(f"{key}: {metrics[key]:.6f}")
    print(f"num_content_targets: {metrics['num_content_targets']}")
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
