from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


@torch.no_grad()
def encode_texts(model, tokenizer, texts, batch_size=16, max_length=128):
    vectors = []
    device = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**enc, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[-1].float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        vectors.append(pooled.cpu())
    return torch.cat(vectors, dim=0).numpy()


def cosine_rows(a, b):
    return np.sum(a * b, axis=1)


def evaluate_pairs(model, tokenizer, pairs, batch_size=16, max_length=128):
    left = [x[0] for x in pairs]
    right = [x[1] for x in pairs]
    gold = np.asarray([float(x[2]) for x in pairs])
    v1 = encode_texts(model, tokenizer, left, batch_size=batch_size, max_length=max_length)
    v2 = encode_texts(model, tokenizer, right, batch_size=batch_size, max_length=max_length)
    pred = cosine_rows(v1, v2)
    rho = spearmanr(pred, gold).statistic
    return {"spearman": float(rho), "num_pairs": int(len(pairs))}


def load_men(path):
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        w1 = re.sub(r"-[nva]$", "", parts[0])
        w2 = re.sub(r"-[nva]$", "", parts[1])
        try:
            score = float(parts[2])
        except ValueError:
            continue
        pairs.append((w1, w2, score))
    return pairs


def load_wordsim353(path):
    import csv
    pairs = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["Word 1"], row["Word 2"], float(row["Human (mean)"])))
    return pairs


def load_simlex(path):
    pairs = []
    with Path(path).open("r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        score_idx = header.index("SimLex999")
        for line in f:
            parts = line.strip().split()
            if len(parts) <= score_idx:
                continue
            pairs.append((parts[0], parts[1], float(parts[score_idx])))
    return pairs
