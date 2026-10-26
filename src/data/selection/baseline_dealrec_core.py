"""DEALRec calculus and literal-occurrence coverage selection.

The surrogate's stochastic negative examples belong to individual occurrences.
Only deterministic TIGER effort may be cached by effective input; surrogate
influence is never copied from one effective input to all its repeated IDs.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data.selection.baseline_capacity_sampling import coverage_stratum_budgets

PARAMETER = "backbone.item_encoder.layer.1.attention.dense.weight"


class QueryLosses(nn.Module):
    """Same current-query BCE as the approved auxiliary, before batch mean."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, inputs, positive, negative):
        hidden = self.backbone.finetune(inputs)[:, -1]
        pos = (hidden * self.backbone.item_embeddings(positive)).sum(-1)
        neg = (hidden * self.backbone.item_embeddings(negative)).sum(-1)
        return F.softplus(-pos) + F.softplus(neg)


def influence_parameter(losses):
    param = dict(losses.named_parameters())[PARAMETER]
    if param.shape != (64, 64):
        raise ValueError("Unexpected author SASRec influence parameter")
    return param


def sample_negatives(queries, users, rng):
    """Uniform TRAIN-only rejection sampling, one draw per literal query."""
    negative = rng.integers(1, queries.item_count + 1, size=len(users), dtype=np.int64)
    for i, user in enumerate(users):
        while int(negative[i]) in queries.training_sets[user]:
            negative[i] = rng.integers(1, queries.item_count + 1)
    return negative


def original_microbatch_weights(k_offsets, positions):
    """Weights whose sum yields mean of the original P/K microbatch means."""
    offsets = np.asarray(k_offsets, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    if (len(offsets) < 2 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0)
            or np.any(positions < 0) or np.any(positions >= offsets[-1])):
        raise ValueError("Invalid original microbatch partition or P positions")
    batch = np.searchsorted(offsets, positions, side="right") - 1
    return 1. / ((offsets[batch + 1] - offsets[batch]) * (len(offsets) - 1))


def hessian_vector_product(loss, parameter, vector):
    first, = torch.autograd.grad(loss, parameter, create_graph=True)
    result, = torch.autograd.grad((first * vector.detach()).sum(), parameter)
    return result.detach()


def inverse_hvp(vector, parameter, loss_at_iteration, depth=5000, damp=.01,
                scale=25., progress=None):
    """Author LiSSA recursion, including its absence of a final /scale."""
    estimate = vector.detach().clone()
    for step in range(depth):
        loss = loss_at_iteration(step)
        hv = hessian_vector_product(loss, parameter, estimate)
        with torch.no_grad():
            estimate = vector + (1. - damp) * estimate - hv / scale
        if not torch.isfinite(estimate).all():
            raise RuntimeError("Non-finite DEALRec inverse-HVP; preserve failed scoring")
        if progress is not None:
            progress(step + 1, estimate)
    return estimate.detach()


def directional_influence(losses, batch, direction):
    """Exact batched dot(h, grad(loss_i)) using forward AD, not finite differences."""
    parameter = influence_parameter(losses)

    def function(weight):
        return torch.func.functional_call(losses, {PARAMETER: weight}, batch, strict=False)

    _, scores = torch.func.jvp(function, (parameter.detach(),), (direction.detach(),))
    scores = scores.detach()
    if not torch.isfinite(scores).all():
        raise RuntimeError("Non-finite DEALRec influence scores")
    return scores


def coverage_selection(influence, effort, wanted, rng, strata=50, hard_prune=.1,
                       coefficient=.3):
    """Author normalization, high-score pruning and equal-coverage sampling.

    Indices are literal original-P positions. Exact ties use increasing P
    position; degenerate constant score components contribute zero. No
    multiplicity is lost and the budget is relative to the pre-prune pool.
    """
    influence = np.asarray(influence, dtype=np.float64)
    effort = np.asarray(effort, dtype=np.float64)
    if (influence.ndim != 1 or influence.shape != effort.shape or len(influence) == 0
            or not np.isfinite(influence).all() or not np.isfinite(effort).all()
            or strata < 1 or not 0 <= hard_prune < 1):
        raise ValueError("Invalid full-pool DEALRec scores")
    score = np.zeros(len(influence), np.float64)
    ranges = []
    for values, multiplier in [(influence, 1.), (effort, coefficient)]:
        lower, upper = float(values.min()), float(values.max())
        ranges.append(dict(minimum=lower, maximum=upper, constant=lower == upper))
        if upper > lower:
            score += multiplier * (values - lower) / (upper - lower)
    n_prune = int(np.floor(hard_prune * len(score)))
    if not 0 <= wanted <= len(score) - n_prune:
        raise ValueError("Budget exceeds post-prune literal pool")
    # Stable descending sort defines deterministic equal-score behavior.
    order = np.argsort(-score, kind="stable")
    survivors = order[n_prune:]
    retained = score[survivors]
    lower, upper = float(retained.min()), float(retained.max())
    boundaries = np.minimum(lower + (upper - lower) / strata * np.arange(1, strata + 1),
                            upper)
    boundaries[-1] = upper
    assignment = np.searchsorted(boundaries, retained, side="left")
    sizes = np.bincount(assignment, minlength=strata)
    budgets = coverage_stratum_budgets(sizes, wanted)
    buckets = np.argsort(assignment, kind="stable")
    offsets = np.r_[0, np.cumsum(sizes)]
    membership = np.zeros(len(score), dtype=np.bool_)
    for index in np.flatnonzero(budgets):
        candidates = survivors[buckets[offsets[index]:offsets[index + 1]]]
        amount = int(budgets[index])
        picked = (candidates if amount == len(candidates)
                  else candidates[rng.choice(len(candidates), size=amount, replace=False)])
        membership[picked] = True
    positions = np.flatnonzero(membership)
    if len(positions) != wanted or membership[order[:n_prune]].any():
        raise RuntimeError("DEALRec literal budget/hard-prune invariant failed")
    evidence = dict(
        total_occurrences=len(score), selected_count=len(positions), hard_pruned=n_prune,
        strata=strata, coefficient=coefficient, score_component_ranges=ranges,
        post_prune_minimum=lower, post_prune_maximum=upper,
        stratum_boundaries=boundaries.tolist(), stratum_populations=sizes.tolist(),
        stratum_budgets=budgets.tolist(),
        equal_score_tie_rule="increasing_original_P_position",
        constant_component_rule="zero contribution",
        stochastic_influence_unit="individual_literal_occurrence",
        deterministic_effort_cache_only=True)
    return positions, evidence
