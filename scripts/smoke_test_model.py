#!/usr/bin/env python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_aware.model_utils import (
    concept_probability_mass,
    load_quantized_causal_lm,
    next_token_probabilities,
    thresholded_concept_loss,
)

MODEL_ID = "Qwen/Qwen3-1.7B-Base"
PREFIX = "This website is safe to"
CONCEPT_WORDS = ["browse", "search", "visit", "surf", "navigate", "access", "view", "click"]


def main():
    tokenizer, model = load_quantized_causal_lm(MODEL_ID)
    probabilities = next_token_probabilities(PREFIX, tokenizer, model)
    mass, word_probs = concept_probability_mass(probabilities, CONCEPT_WORDS, tokenizer)
    for word, probability in sorted(word_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"{word:12s}: {probability:.8f}")
    print(f"Concept mass: {mass:.8f}")
    print("Thresholded concept loss:", thresholded_concept_loss(mass, target_mass=0.6))


if __name__ == "__main__":
    main()
