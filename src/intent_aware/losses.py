from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_ce(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask=None):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    if attention_mask is not None:
        valid = attention_mask[:, 1:].bool()
        shift_labels = shift_labels.masked_fill(~valid, -100)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def concept_terms_from_logits(
    logits: torch.Tensor,
    targets_batch: list[list[dict]],
    *,
    target_mass: float = 0.6,
    max_targets_per_sequence: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return Zhang-style thresholded mass loss and context-preservation KL.

    The optional KL uses a stored base-model distribution (`concept_prior`) over
    the concept set. It compares that prior with the fine-tuned model's
    distribution normalized *within the same set*. This preserves contextual
    preferences while still allowing the total concept mass to increase.
    """
    mass_losses = []
    kl_losses = []
    masses = []
    for b, targets in enumerate(targets_batch):
        chosen = targets[:max_targets_per_sequence] if max_targets_per_sequence else targets
        for target in chosen:
            pos = int(target["position"])
            ids = [int(x) for x in target.get("concept_token_ids", [])]
            if pos <= 0 or pos >= logits.shape[1] or not ids:
                continue
            log_probs = F.log_softmax(logits[b, pos - 1].float(), dim=-1)
            ids_t = torch.tensor(ids, device=logits.device, dtype=torch.long)
            set_log_probs = log_probs.index_select(0, ids_t)
            log_mass = torch.logsumexp(set_log_probs, dim=0)
            mass = log_mass.exp()
            masses.append(mass.detach())
            mass_losses.append(torch.where(mass >= target_mass, torch.zeros_like(log_mass), -log_mass))

            prior = target.get("concept_prior")
            if prior and len(prior) == len(ids):
                q = torch.tensor(prior, device=logits.device, dtype=torch.float32)
                q = q / q.sum().clamp(min=1e-12)
                log_r = set_log_probs - log_mass
                kl_losses.append(torch.sum(q * (torch.log(q.clamp(min=1e-12)) - log_r)))

    zero = logits.sum() * 0.0
    mass_loss = torch.stack(mass_losses).mean() if mass_losses else zero
    kl_loss = torch.stack(kl_losses).mean() if kl_losses else zero
    if masses:
        masses_t = torch.stack(masses)
        stats = {
            "num_targets": int(masses_t.numel()),
            "mean_mass": float(masses_t.mean().item()),
            "threshold_hit_rate": float((masses_t >= target_mass).float().mean().item()),
        }
    else:
        stats = {"num_targets": 0, "mean_mass": float("nan"), "threshold_hit_rate": float("nan")}
    return mass_loss, kl_loss, stats


def combined_intent_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    targets_batch: list[list[dict]],
    *,
    attention_mask=None,
    concept_weight: float = 0.5,
    target_mass: float = 0.6,
    preserve_beta: float = 0.0,
    max_targets_per_sequence: int | None = None,
):
    ce = causal_lm_ce(logits, input_ids, attention_mask)
    concept, preserve_kl, stats = concept_terms_from_logits(
        logits,
        targets_batch,
        target_mass=target_mass,
        max_targets_per_sequence=max_targets_per_sequence,
    )
    total = (1.0 - concept_weight) * ce + concept_weight * concept + preserve_beta * preserve_kl
    return total, ce.detach(), concept.detach(), preserve_kl.detach(), stats
