from __future__ import annotations

from typing import Iterable

TARGET_POS = {"NOUN", "VERB", "ADJ"}
COMPOUND_CHARS = {"-", "–", "—", "'", "’"}


def extract_content_words(text: str, nlp, tokenizer, max_lm_tokens: int = 256) -> list[dict]:
    """Extract clean NOUN/VERB/ADJ targets aligned to exactly one LM token.

    Alignment is checked in the original sentence through tokenizer character
    offsets. This avoids relying on standalone word tokenization.
    """
    doc = nlp(text)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_lm_tokens,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    targets: list[dict] = []

    for tok in doc:
        if tok.pos_ not in TARGET_POS or not tok.is_alpha or len(tok.text) < 2:
            continue

        start = tok.idx
        end = start + len(tok.text)
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        if before in COMPOUND_CHARS or after in COMPOUND_CHARS:
            continue

        overlapping = []
        for idx, (s, e) in enumerate(offsets):
            if s == e:
                continue
            if s < end and e > start:
                overlapping.append(idx)

        if len(overlapping) != 1:
            continue
        pos = overlapping[0]
        if pos <= 0 or pos >= max_lm_tokens:
            continue

        token_start, token_end = offsets[pos]
        if text[token_start:start].strip() or text[end:token_end].strip():
            continue

        targets.append(
            {
                "word": tok.text,
                "lemma": tok.lemma_,
                "pos": tok.pos_,
                "position": int(pos),
                "token_id": int(input_ids[pos]),
                "char_start": int(start),
                "char_end": int(end),
            }
        )
    return targets


def keep_reasonable_texts(
    rows: Iterable[dict],
    *,
    min_chars: int = 50,
    max_chars: int = 2000,
    max_samples: int = 2000,
):
    count = 0
    for row in rows:
        text = row.get("text", "")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not (min_chars < len(text) < max_chars):
            continue
        yield text
        count += 1
        if count >= max_samples:
            break
