from typing import Dict, Iterable, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def load_quantized_causal_lm(
    model_id: str,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load a causal language model in 4-bit NF4 quantization.

    Intended primarily for development/training on limited-memory GPUs
    such as the Tesla T4.
    """

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )

    model.eval()

    return tokenizer, model


def get_single_token_id(
    word: str,
    tokenizer,
) -> Optional[int]:
    """
    Return the token id for a word preceded by a space if the word
    corresponds to exactly one tokenizer token.

    Returns None for multi-token words.
    """

    token_ids = tokenizer.encode(
        " " + word.strip(),
        add_special_tokens=False,
    )

    if len(token_ids) != 1:
        return None

    return token_ids[0]


@torch.no_grad()
def next_token_probabilities(
    prefix: str,
    tokenizer,
    model,
) -> torch.Tensor:
    """
    Compute P(next_token | prefix) over the full vocabulary.
    """

    inputs = tokenizer(
        prefix,
        return_tensors="pt",
    ).to(model.device)

    outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    probabilities = torch.softmax(
        logits.float(),
        dim=-1,
    )

    return probabilities


def concept_probability_mass(
    probabilities: torch.Tensor,
    words: Iterable[str],
    tokenizer,
) -> Tuple[float, Dict[str, float]]:
    """
    Sum model probability assigned to valid single-token members
    of a candidate concept set.
    """

    word_probabilities = {}

    for word in words:
        token_id = get_single_token_id(
            word,
            tokenizer,
        )

        if token_id is None:
            continue

        probability = float(
            probabilities[token_id].item()
        )

        word_probabilities[word] = probability

    total_mass = sum(
        word_probabilities.values()
    )

    return total_mass, word_probabilities


def thresholded_concept_loss(
    concept_mass: float,
    target_mass: float = 0.6,
    eps: float = 1e-9,
) -> float:
    """
    Zhang-style thresholded concept loss.

    If the model already assigns at least target_mass probability
    to the concept set, no additional concept pressure is applied.
    """

    if concept_mass >= target_mass:
        return 0.0

    return float(
        -torch.log(
            torch.tensor(
                max(concept_mass, eps),
                dtype=torch.float32,
            )
        ).item()
    )