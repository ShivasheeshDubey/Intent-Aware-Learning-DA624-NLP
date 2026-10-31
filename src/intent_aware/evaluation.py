from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .io_utils import read_jsonl


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_length=256, max_samples=None):
        self.items = []
        for row in read_jsonl(path):
            enc = tokenizer(
                row["text"],
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            targets = [t for t in row.get("targets", []) if 0 < int(t["position"]) < ids.numel()]
            self.items.append({"input_ids": ids, "attention_mask": mask, "targets": targets})
            if max_samples and len(self.items) >= max_samples:
                break

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _collate(batch, pad_token_id):
    ids = pad_sequence([x["input_ids"] for x in batch], batch_first=True, padding_value=pad_token_id)
    masks = pad_sequence([x["attention_mask"] for x in batch], batch_first=True, padding_value=0)
    return {"input_ids": ids, "attention_mask": masks, "targets": [x["targets"] for x in batch]}


@torch.no_grad()
def evaluate_model(model, tokenizer, dataset, batch_size=2):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
    )
    global_nll = []
    content_nll = []
    content_correct = []
    concept_masses = []

    device = next(model.parameters()).device
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
        labels = ids[:, 1:]
        valid = mask[:, 1:].bool()
        token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        global_nll.extend((-token_lp[valid]).cpu().tolist())

        for b, targets in enumerate(batch["targets"]):
            for t in targets:
                pos = int(t["position"])
                if not (0 < pos < ids.shape[1]) or not bool(mask[b, pos]):
                    continue
                lp = F.log_softmax(logits[b, pos - 1].float(), dim=-1)
                true_id = int(ids[b, pos].item())
                content_nll.append(float(-lp[true_id].item()))
                content_correct.append(int(lp.argmax().item() == true_id))
                cids = [int(x) for x in t.get("concept_token_ids", [])]
                if cids:
                    cids_t = torch.tensor(cids, device=lp.device, dtype=torch.long)
                    mass = torch.logsumexp(lp.index_select(0, cids_t), dim=0).exp().item()
                    concept_masses.append(float(mass))

    result = {
        "global_nll": float(np.mean(global_nll)),
        "global_perplexity": float(math.exp(np.mean(global_nll))),
        "content_nll": float(np.mean(content_nll)) if content_nll else float("nan"),
        "content_perplexity": float(math.exp(np.mean(content_nll))) if content_nll else float("nan"),
        "content_accuracy": float(np.mean(content_correct)) if content_correct else float("nan"),
        "mean_concept_mass": float(np.mean(concept_masses)) if concept_masses else float("nan"),
        "num_tokens": len(global_nll),
        "num_content_targets": len(content_nll),
        "per_target_nll": content_nll,
        "per_target_correct": content_correct,
    }
    return result
