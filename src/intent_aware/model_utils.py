from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def make_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_quantized_causal_lm(
    model_id: str,
    *,
    train: bool = False,
    device_map: str = "auto",
):
    tokenizer = load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=make_4bit_config(),
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    if train:
        model.train()
    else:
        model.eval()
    return tokenizer, model


def load_model_with_adapter(
    base_model_id: str,
    adapter_path: Optional[str] = None,
    *,
    device_map: str = "auto",
):
    from peft import PeftModel

    tokenizer, model = load_quantized_causal_lm(
        base_model_id, train=False, device_map=device_map
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
    return tokenizer, model


def get_single_token_id(word: str, tokenizer) -> Optional[int]:
    ids = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


@torch.no_grad()
def next_token_probabilities(prefix: str, tokenizer, model) -> torch.Tensor:
    inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1]
    return torch.softmax(logits.float(), dim=-1)


def concept_probability_mass(
    probabilities: torch.Tensor,
    words: Iterable[str],
    tokenizer,
) -> Tuple[float, Dict[str, float]]:
    word_probabilities: Dict[str, float] = {}
    for word in words:
        token_id = get_single_token_id(word, tokenizer)
        if token_id is None:
            continue
        word_probabilities[word] = float(probabilities[token_id].item())
    return sum(word_probabilities.values()), word_probabilities


def thresholded_concept_loss(
    concept_mass: float,
    target_mass: float = 0.6,
    eps: float = 1e-9,
) -> float:
    if concept_mass >= target_mass:
        return 0.0
    return float(-torch.log(torch.tensor(max(concept_mass, eps))).item())
