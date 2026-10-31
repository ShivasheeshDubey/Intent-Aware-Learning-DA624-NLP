import torch

from intent_aware.concept_sets import parse_allowed_list
from intent_aware.losses import concept_terms_from_logits


def test_parse_allowed_list_filters_hallucinations():
    out = parse_allowed_list('["browse", "invented", "surf"]', ["browse", "surf"])
    assert out == ["browse", "surf"]


def test_thresholded_concept_loss_zero_above_mass():
    # vocab=3, probabilities approximately [0.7, 0.2, 0.1]
    logits = torch.log(torch.tensor([[[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]]]))
    targets = [[{"position": 1, "concept_token_ids": [0], "concept_prior": [1.0]}]]
    mass_loss, kl, stats = concept_terms_from_logits(logits, targets, target_mass=0.6)
    assert abs(float(mass_loss.item())) < 1e-7
    assert stats["threshold_hit_rate"] == 1.0


def test_concept_loss_positive_below_mass():
    logits = torch.log(torch.tensor([[[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]]))
    targets = [[{"position": 1, "concept_token_ids": [0], "concept_prior": [1.0]}]]
    mass_loss, _, stats = concept_terms_from_logits(logits, targets, target_mass=0.6)
    assert float(mass_loss.item()) > 0
    assert stats["threshold_hit_rate"] == 0.0
