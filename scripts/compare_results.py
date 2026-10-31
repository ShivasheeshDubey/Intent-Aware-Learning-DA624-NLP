#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_aware.stats import paired_bootstrap_difference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--concept", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    concept = json.loads(Path(args.concept).read_text(encoding="utf-8"))
    result = {
        "content_nll_difference_concept_minus_base": paired_bootstrap_difference(
            base["per_target_nll"], concept["per_target_nll"], n_boot=args.n_boot, seed=args.seed
        ),
        "content_accuracy_difference_concept_minus_base": paired_bootstrap_difference(
            base["per_target_correct"], concept["per_target_correct"], n_boot=args.n_boot, seed=args.seed
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
