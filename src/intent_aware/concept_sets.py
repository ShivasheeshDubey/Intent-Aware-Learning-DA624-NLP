from __future__ import annotations

import ast
import json
import re
from typing import Iterable

import torch


def decode_candidate_token(tokenizer, token_id: int) -> str | None:
    raw = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    if not raw.startswith(" "):
        return None
    word = raw.strip()
    if len(word) < 2 or not word.isalpha():
        return None
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) != 1 or int(ids[0]) != int(token_id):
        return None
    return word


def topk_word_candidates(
    logits_at_position: torch.Tensor,
    tokenizer,
    *,
    top_k: int = 100,
    max_candidates: int = 40,
) -> list[dict]:
    """Return whole-word single-token candidates from top-k next-token logits."""
    probs = torch.softmax(logits_at_position.float(), dim=-1)
    k = min(top_k, probs.shape[-1])
    values, ids = torch.topk(probs, k=k)
    out = []
    seen = set()
    for prob, token_id in zip(values.tolist(), ids.tolist()):
        word = decode_candidate_token(tokenizer, token_id)
        if not word:
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"word": word, "token_id": int(token_id), "prob": float(prob)})
        if len(out) >= max_candidates:
            break
    return out


def build_filter_prompt(text: str, target: str, candidates: Iterable[str]) -> str:
    candidates = list(candidates)
    return (
        "You are filtering contextual synonyms for a language-model training dataset.\n"
        "A valid candidate must be able to replace the target word in THIS exact text "
        "without materially changing the intended meaning or grammatical role.\n\n"
        f"TEXT:\n{text}\n\n"
        f"TARGET: {target}\n"
        f"ALLOWED CANDIDATES: {json.dumps(candidates)}\n\n"
        "Return ONLY a JSON array containing zero or more strings copied exactly from "
        "ALLOWED CANDIDATES. Do not explain."
    )


def parse_allowed_list(response: str, allowed: Iterable[str]) -> list[str]:
    allowed_map = {w.lower(): w for w in allowed}
    match = re.search(r"\[[\s\S]*?\]", response)
    if not match:
        return []
    text = match.group(0)
    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
    if not isinstance(parsed, list):
        return []
    out = []
    seen = set()
    for item in parsed:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        if key in allowed_map and key not in seen:
            seen.add(key)
            out.append(allowed_map[key])
    return out
