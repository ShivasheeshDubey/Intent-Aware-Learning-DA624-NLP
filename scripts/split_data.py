#!/usr/bin/env python
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_aware.io_utils import read_jsonl, write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = list(read_jsonl(args.input))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    splits = {
        "train": rows[:n_train],
        "val": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val:],
    }
    out = Path(args.output_dir)
    for name, data in splits.items():
        path = out / f"{name}.jsonl"
        write_jsonl(path, data)
        print(f"{name}: {len(data)} -> {path}")


if __name__ == "__main__":
    main()
