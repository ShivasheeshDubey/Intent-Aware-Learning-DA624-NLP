#!/usr/bin/env python
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments

from intent_aware.model_utils import load_quantized_causal_lm
from intent_aware.training import ConceptCollator, ConceptDataset, IntentAwareTrainer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--concept-weight", type=float, default=0.5)
    p.add_argument("--target-mass", type=float, default=0.6)
    p.add_argument("--preserve-beta", type=float, default=0.0, help="Context-preservation KL coefficient; 0 reproduces the base concept objective")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-targets-per-sequence", type=int, default=4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=7e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=4)
    p.add_argument("--lora-dropout", type=float, default=0.2)
    args = p.parse_args()

    seed_all(args.seed)
    tokenizer, model = load_quantized_causal_lm(args.model, train=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = ConceptDataset(args.train, tokenizer, args.max_length, args.max_samples)
    collator = ConceptCollator(tokenizer.pad_token_id)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = IntentAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
        concept_weight=args.concept_weight,
        target_mass=args.target_mass,
        preserve_beta=args.preserve_beta,
        max_targets_per_sequence=args.max_targets_per_sequence,
    )
    trainer.train()
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter to {out}")


if __name__ == "__main__":
    main()
