from __future__ import annotations

from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import Trainer

from .io_utils import read_jsonl
from .losses import combined_intent_loss


class ConceptDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256, max_samples=None):
        self.items = []
        for row in read_jsonl(path):
            text = row["text"]
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            targets = []
            for t in row.get("targets", []):
                pos = int(t["position"])
                if 0 < pos < ids.numel() and t.get("concept_token_ids"):
                    targets.append(t)
            if targets:
                self.items.append({"input_ids": ids, "attention_mask": mask, "targets": targets})
            if max_samples and len(self.items) >= max_samples:
                break
        if not self.items:
            raise ValueError(f"No usable training samples found in {Path(path)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class ConceptCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features):
        ids = pad_sequence(
            [x["input_ids"] for x in features],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        masks = pad_sequence(
            [x["attention_mask"] for x in features],
            batch_first=True,
            padding_value=0,
        )
        return {
            "input_ids": ids,
            "attention_mask": masks,
            "targets": [x["targets"] for x in features],
        }


class IntentAwareTrainer(Trainer):
    def __init__(
        self,
        *args,
        concept_weight=0.5,
        target_mass=0.6,
        max_targets_per_sequence=None,
        preserve_beta=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.concept_weight = float(concept_weight)
        self.target_mass = float(target_mass)
        self.max_targets_per_sequence = max_targets_per_sequence
        self.preserve_beta = float(preserve_beta)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        targets = inputs.pop("targets")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        loss, ce, concept, preserve_kl, _ = combined_intent_loss(
            outputs.logits,
            inputs["input_ids"],
            targets,
            attention_mask=inputs.get("attention_mask"),
            concept_weight=self.concept_weight,
            target_mass=self.target_mass,
            preserve_beta=self.preserve_beta,
            max_targets_per_sequence=self.max_targets_per_sequence,
        )
        self.log({"ce_loss": float(ce.item()), "concept_loss": float(concept.item()), "preserve_kl": float(preserve_kl.item())})
        return (loss, outputs) if return_outputs else loss
