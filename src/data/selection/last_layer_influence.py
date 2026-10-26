from __future__ import annotations

import torch


def last_layer_gradient_similarity(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    reference_gradients: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Cosine similarity between per-example and reference last-layer gradients.

    Shapes are ``hidden_states=[B, L, H]``, ``logits=[B, L, V]``,
    ``labels=[B, L]``, and ``reference_gradients=[L, V, H]``. The
    per-example outer products are never materialized.
    """
    if hidden_states.ndim != 3 or logits.ndim != 3:
        raise ValueError("hidden_states and logits must be three-dimensional")
    if labels.ndim != 2 or reference_gradients.ndim != 3:
        raise ValueError("labels and reference_gradients have invalid shape")
    batch_size, levels, hidden_dim = hidden_states.shape
    if logits.shape[:2] != (batch_size, levels):
        raise ValueError("logits must align with hidden_states")
    vocab_size = logits.shape[2]
    if labels.shape != (batch_size, levels):
        raise ValueError("labels must have shape [B, L]")
    if reference_gradients.shape != (levels, vocab_size, hidden_dim):
        raise ValueError("reference_gradients must have shape [L, V, H]")

    errors = torch.softmax(logits.float(), dim=-1)
    errors.scatter_add_(
        2,
        labels.long().unsqueeze(-1),
        -torch.ones_like(labels, dtype=errors.dtype).unsqueeze(-1),
    )
    dots = torch.einsum(
        "blv,lvh,blh->b", errors, reference_gradients.float(), hidden_states.float()
    )
    candidate_norm_sq = (
        errors.square().sum(dim=-1) * hidden_states.float().square().sum(dim=-1)
    ).sum(dim=-1)
    reference_norm = torch.linalg.vector_norm(reference_gradients.float())
    denominator = candidate_norm_sq.clamp_min(eps).sqrt() * reference_norm.clamp_min(
        eps
    )
    return dots / denominator
