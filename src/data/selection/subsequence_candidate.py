from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset


FORMAT_VERSION = 1
SELECTION_FORMAT_VERSION = 1


def _as_flat_int_tensor(value: torch.Tensor | Sequence[int]) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.int32).flatten().contiguous()


def _validate_budget(num_candidates: int, budget: int) -> None:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if budget > num_candidates:
        raise ValueError(
            f"budget {budget} exceeds candidate count {num_candidates}"
        )


def random_candidate_indices(
    num_candidates: int,
    budget: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Uniformly sample B global candidate indices without replacement."""
    _validate_budget(num_candidates, budget)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(num_candidates, generator=generator)[:budget]


def weighted_candidate_indices(
    weights: torch.Tensor,
    budget: int,
    *,
    seed: int,
    excluded_indices: Optional[torch.Tensor | Sequence[int]] = None,
) -> torch.Tensor:
    """Sample unique indices proportionally to positive weights via Gumbel Top-k."""
    flat_weights = torch.as_tensor(weights, dtype=torch.float64).flatten().cpu()
    _validate_budget(flat_weights.numel(), budget)
    if not torch.isfinite(flat_weights).all() or torch.any(flat_weights <= 0):
        raise ValueError("weights must all be positive and finite")

    excluded = torch.as_tensor(
        [] if excluded_indices is None else excluded_indices, dtype=torch.int64
    ).flatten()
    if excluded.numel() and (
        int(excluded.min()) < 0 or int(excluded.max()) >= flat_weights.numel()
    ):
        raise ValueError("excluded index is outside the candidate pool")
    if torch.unique(excluded).numel() != excluded.numel():
        raise ValueError("excluded indices must be unique")
    if budget > flat_weights.numel() - excluded.numel():
        raise ValueError("budget exceeds the number of non-excluded candidates")
    if budget == 0:
        return torch.empty(0, dtype=torch.int64)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    uniform = torch.rand(
        flat_weights.numel(), generator=generator, dtype=torch.float64
    ).clamp_(
        min=torch.finfo(torch.float64).tiny,
        max=1.0 - torch.finfo(torch.float64).eps,
    )
    keys = torch.log(flat_weights) - torch.log(-torch.log(uniform))
    keys[excluded] = -torch.inf
    return torch.topk(keys, k=budget, sorted=False).indices


def percentile_rank_scores(scores: torch.Tensor) -> torch.Tensor:
    """Map finite scores to average percentile ranks with stable tie handling."""
    flat = torch.as_tensor(scores).flatten().to(dtype=torch.float64, device="cpu")
    if not flat.numel():
        raise ValueError("scores must be non-empty")
    if not torch.isfinite(flat).all():
        raise ValueError("scores must all be finite")
    if flat.numel() == 1:
        return torch.full((1,), 0.5, dtype=torch.float32)

    sorted_values, order = torch.sort(flat, stable=True)
    first_in_group = torch.ones(sorted_values.numel(), dtype=torch.bool)
    first_in_group[1:] = sorted_values[1:] != sorted_values[:-1]
    group_ids = torch.cumsum(first_in_group.to(torch.int64), dim=0) - 1
    positions = torch.arange(sorted_values.numel(), dtype=torch.float64)
    group_count = int(group_ids[-1]) + 1
    sums = torch.zeros(group_count, dtype=torch.float64)
    counts = torch.zeros(group_count, dtype=torch.float64)
    sums.scatter_add_(0, group_ids, positions)
    counts.scatter_add_(0, group_ids, torch.ones_like(positions))
    sorted_ranks = (sums / counts)[group_ids] / (flat.numel() - 1)
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks.to(torch.float32)


def validation_aligned_difficulty_scores(
    validation_alignment: torch.Tensor,
    current_loss: torch.Tensor,
    *,
    alignment_weight: float = 0.5,
) -> torch.Tensor:
    """Fuse validation-gradient alignment and difficulty by robust percentile rank."""
    if not 0 <= alignment_weight <= 1:
        raise ValueError("alignment_weight must be between zero and one")
    alignment = torch.as_tensor(validation_alignment).flatten().cpu()
    losses = torch.as_tensor(current_loss).flatten().cpu()
    if alignment.shape != losses.shape:
        raise ValueError("validation_alignment and current_loss must have equal shape")
    alignment_rank = percentile_rank_scores(alignment)
    loss_rank = percentile_rank_scores(losses)
    return alignment_weight * alignment_rank + (1 - alignment_weight) * loss_rank


def coverage_first_candidate_indices(
    candidate_triples: torch.Tensor,
    budget: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Maximize user coverage, then uniformly fill the remaining budget."""
    triples = torch.as_tensor(candidate_triples)
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("candidate_triples must have shape [N, 3]")
    num_candidates = int(triples.shape[0])
    _validate_budget(num_candidates, budget)
    if budget == 0:
        return torch.empty(0, dtype=torch.int64)

    user_rows = triples[:, 0].to(dtype=torch.int64, device="cpu")
    if int(user_rows.min()) < 0:
        raise ValueError("candidate user rows must be non-negative")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    # Stable grouping after a random permutation makes the first candidate in
    # each user group a uniform random representative for that user.
    candidate_order = torch.randperm(num_candidates, generator=generator)
    permuted_users = user_rows[candidate_order]
    grouped_positions = torch.argsort(permuted_users, stable=True)
    grouped_users = permuted_users[grouped_positions]
    first_in_group = torch.ones(num_candidates, dtype=torch.bool)
    first_in_group[1:] = grouped_users[1:] != grouped_users[:-1]
    representatives = candidate_order[grouped_positions[first_in_group]]

    if budget <= representatives.numel():
        selected = representatives[
            torch.randperm(representatives.numel(), generator=generator)[:budget]
        ]
    else:
        is_remaining = torch.ones(num_candidates, dtype=torch.bool)
        is_remaining[representatives] = False
        remaining = torch.arange(num_candidates, dtype=torch.int64)[is_remaining]
        fill = remaining[
            torch.randperm(remaining.numel(), generator=generator)[
                : budget - representatives.numel()
            ]
        ]
        selected = torch.cat((representatives, fill))

    return selected[torch.randperm(selected.numel(), generator=generator)]


def latest_coverage_candidate_indices(
    candidate_triples: torch.Tensor,
    budget: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Cover users with their latest, longest-context candidate, then fill."""
    triples = torch.as_tensor(candidate_triples).to(device="cpu")
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("candidate_triples must have shape [N, 3]")
    num_candidates = int(triples.shape[0])
    _validate_budget(num_candidates, budget)
    if budget == 0:
        return torch.empty(0, dtype=torch.int64)

    user_rows, starts, targets = triples.to(dtype=torch.int64).unbind(dim=1)
    if int(user_rows.min()) < 0:
        raise ValueError("candidate user rows must be non-negative")

    # Stable lexicographic ordering by user ascending, target descending, then
    # start ascending makes the first row per user the latest target with the
    # longest available history.
    ordered = torch.arange(num_candidates, dtype=torch.int64)
    ordered = ordered[torch.argsort(starts[ordered], stable=True)]
    ordered = ordered[
        torch.argsort(targets[ordered], descending=True, stable=True)
    ]
    ordered = ordered[torch.argsort(user_rows[ordered], stable=True)]
    ordered_users = user_rows[ordered]
    first_in_user = torch.ones(num_candidates, dtype=torch.bool)
    first_in_user[1:] = ordered_users[1:] != ordered_users[:-1]
    anchors = ordered[first_in_user]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if budget <= anchors.numel():
        selected = anchors[
            torch.randperm(anchors.numel(), generator=generator)[:budget]
        ]
    else:
        is_remaining = torch.ones(num_candidates, dtype=torch.bool)
        is_remaining[anchors] = False
        remaining = torch.arange(num_candidates, dtype=torch.int64)[is_remaining]
        fill = remaining[
            torch.randperm(remaining.numel(), generator=generator)[
                : budget - anchors.numel()
            ]
        ]
        selected = torch.cat((anchors, fill))

    return selected[torch.randperm(selected.numel(), generator=generator)]


def multi_target_longest_candidate_indices(
    candidate_triples: torch.Tensor,
    *,
    max_items: int,
) -> torch.Tensor:
    """Keep the single longest effective context for every user-target pair."""
    triples = torch.as_tensor(candidate_triples).to(device="cpu")
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("candidate_triples must have shape [N, 3]")
    if max_items < 2:
        raise ValueError("max_items must fit a history item and target")
    if not triples.shape[0]:
        return torch.empty(0, dtype=torch.int64)

    _, starts, targets = triples.to(dtype=torch.int64).unbind(dim=1)
    minimum_starts = torch.clamp(targets + 1 - max_items, min=0)
    return torch.nonzero(starts == minimum_starts, as_tuple=False).flatten()


def user_balanced_top_b_candidate_indices(
    candidate_triples: torch.Tensor,
    scores: torch.Tensor,
    budget: int,
    *,
    anchors: Optional[torch.Tensor | Sequence[int]] = None,
) -> torch.Tensor:
    """Select score-ranked candidates in user-balanced round-robin order."""
    triples = torch.as_tensor(candidate_triples).to(device="cpu")
    flat_scores = torch.as_tensor(scores).flatten().to(device="cpu")
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("candidate_triples must have shape [N, 3]")
    num_candidates = int(triples.shape[0])
    if flat_scores.numel() != num_candidates:
        raise ValueError("scores must contain one value per candidate")
    if not torch.isfinite(flat_scores).all():
        raise ValueError("scores must all be finite")
    _validate_budget(num_candidates, budget)

    selected_anchors = torch.as_tensor(
        [] if anchors is None else anchors, dtype=torch.int64
    ).flatten()
    if selected_anchors.numel() > budget:
        raise ValueError("anchor count exceeds budget")
    if selected_anchors.numel() and (
        int(selected_anchors.min()) < 0
        or int(selected_anchors.max()) >= num_candidates
    ):
        raise ValueError("anchor index is outside the candidate pool")
    if torch.unique(selected_anchors).numel() != selected_anchors.numel():
        raise ValueError("anchor indices must be unique")
    if selected_anchors.numel() == budget:
        return selected_anchors.clone()

    is_remaining = torch.ones(num_candidates, dtype=torch.bool)
    is_remaining[selected_anchors] = False
    remaining = torch.arange(num_candidates, dtype=torch.int64)[is_remaining]
    remaining_users = triples[remaining, 0].to(dtype=torch.int64)
    remaining_scores = flat_scores[remaining]

    # First group candidates by user while preserving descending score order.
    score_order = torch.argsort(remaining_scores, descending=True, stable=True)
    score_users = remaining_users[score_order]
    grouped_positions = torch.argsort(score_users, stable=True)
    grouped_remaining_positions = score_order[grouped_positions]
    grouped_users = remaining_users[grouped_remaining_positions]

    first_in_user = torch.ones(grouped_users.numel(), dtype=torch.bool)
    first_in_user[1:] = grouped_users[1:] != grouped_users[:-1]
    group_starts = torch.nonzero(first_in_user, as_tuple=False).flatten()
    group_ends = torch.cat(
        (group_starts[1:], torch.tensor([grouped_users.numel()]))
    )
    group_lengths = group_ends - group_starts
    within_user_rank = torch.arange(grouped_users.numel()) - torch.repeat_interleave(
        group_starts, group_lengths
    )

    # Reorder by within-user rank while preserving the global descending-score
    # order inside every round.  This makes a partial round choose the strongest
    # users' next candidates rather than favoring lower numeric user IDs.
    rank_by_remaining_position = torch.empty_like(within_user_rank)
    rank_by_remaining_position[grouped_remaining_positions] = within_user_rank
    round_order = torch.argsort(
        rank_by_remaining_position[score_order], stable=True
    )
    balanced_remaining = remaining[score_order[round_order]]
    needed = budget - selected_anchors.numel()
    return torch.cat((selected_anchors, balanced_remaining[:needed]))


def adaptive_rho_replacement_fraction(
    scores: torch.Tensor,
    *,
    positive_scale: float = 0.5,
    max_fraction: float = 0.2,
) -> float:
    """Map the remaining positive-utility mass to a bounded replacement rate."""
    flat_scores = torch.as_tensor(scores).flatten().cpu()
    if not flat_scores.numel():
        raise ValueError("scores must be non-empty")
    if not torch.isfinite(flat_scores).all():
        raise ValueError("scores must all be finite")
    if positive_scale < 0:
        raise ValueError("positive_scale must be non-negative")
    if not 0 <= max_fraction <= 1:
        raise ValueError("max_fraction must be between zero and one")
    positive_fraction = float((flat_scores > 0).to(torch.float32).mean())
    return min(max_fraction, positive_scale * positive_fraction)


def horizon_adjusted_rho_gain(
    scores: torch.Tensor,
    *,
    remaining_updates: int,
    scoring_cost: float = 1.0,
) -> float:
    """Estimate refinable positive utility per scoring cost over the runway."""
    flat_scores = torch.as_tensor(scores).flatten().cpu()
    if not flat_scores.numel() or not torch.isfinite(flat_scores).all():
        raise ValueError("scores must be non-empty and finite")
    if remaining_updates < 0:
        raise ValueError("remaining_updates must be non-negative")
    if scoring_cost <= 0:
        raise ValueError("scoring_cost must be positive")
    positive_mass = float(flat_scores.clamp_min(0).mean())
    return positive_mass * remaining_updates / scoring_cost


def rho_refinement_is_worthwhile(
    current_scores: torch.Tensor,
    previous_scores: torch.Tensor,
    *,
    remaining_updates: int,
    previous_remaining_updates: int,
    min_gain_ratio: float = 0.5,
) -> tuple[bool, float]:
    """Gate another refinement by its horizon-adjusted gain versus the prior one."""
    if min_gain_ratio < 0:
        raise ValueError("min_gain_ratio must be non-negative")
    current_gain = horizon_adjusted_rho_gain(
        current_scores, remaining_updates=remaining_updates
    )
    previous_gain = horizon_adjusted_rho_gain(
        previous_scores, remaining_updates=previous_remaining_updates
    )
    ratio = current_gain / previous_gain if previous_gain > 0 else 0.0
    return ratio >= min_gain_ratio, ratio


def stagewise_rho_replacement_candidate_indices(
    candidate_triples: torch.Tensor,
    scores: torch.Tensor,
    current_indices: torch.Tensor | Sequence[int],
    *,
    replacement_fraction: float,
    min_standardized_swap_margin: float = 0.05,
    seed: int,
) -> torch.Tensor:
    """Preserve anchors and execute only profitable structure-matched swaps.

    ``replacement_fraction`` is an upper bound.  Within each structural bin,
    the weakest current candidates are paired with user-balanced high-scoring
    unseen challengers.  A pair is swapped only when its score margin exceeds
    ``min_standardized_swap_margin`` times the population standard deviation
    of all proposed margins.  Missing coverage anchors remain mandatory.
    """
    triples = torch.as_tensor(candidate_triples).to(dtype=torch.int64, device="cpu")
    flat_scores = torch.as_tensor(scores).flatten().cpu()
    current = torch.as_tensor(current_indices, dtype=torch.int64).flatten().cpu()
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("candidate_triples must have shape [N, 3]")
    if flat_scores.numel() != triples.shape[0]:
        raise ValueError("scores must contain one value per candidate")
    if not torch.isfinite(flat_scores).all():
        raise ValueError("scores must all be finite")
    if not 0 <= replacement_fraction <= 1:
        raise ValueError("replacement_fraction must be between zero and one")
    if min_standardized_swap_margin < 0:
        raise ValueError("min_standardized_swap_margin must be non-negative")
    if current.numel() and (
        int(current.min()) < 0 or int(current.max()) >= triples.shape[0]
    ):
        raise ValueError("current index is outside the candidate pool")
    if torch.unique(current).numel() != current.numel():
        raise ValueError("current indices must be unique")

    num_users = int(torch.unique(triples[:, 0]).numel())
    if current.numel() < num_users:
        raise ValueError("current selection cannot preserve every user anchor")
    anchors = latest_coverage_candidate_indices(triples, num_users, seed=seed)
    is_current = torch.zeros(triples.shape[0], dtype=torch.bool)
    is_current[current] = True
    is_anchor = torch.zeros(triples.shape[0], dtype=torch.bool)
    is_anchor[anchors] = True
    residual = current[~is_anchor[current]]
    missing_anchors = anchors[~is_current[anchors]]
    refinable_budget = current.numel() - anchors.numel()
    replacement_total = round(refinable_budget * replacement_fraction)
    if replacement_total == 0 and missing_anchors.numel() == 0:
        return current.clone()

    histories = triples[:, 2] - triples[:, 1]
    target_bins = torch.floor(
        torch.log2((triples[:, 2] + 1).to(torch.float64))
    ).to(torch.int64)
    bin_ids = histories * (int(target_bins.max()) + 1) + target_bins
    displaced_parts: list[torch.Tensor] = []
    if missing_anchors.numel():
        unmatched_missing_parts: list[torch.Tensor] = []
        for bin_id in torch.unique(bin_ids[missing_anchors], sorted=True).tolist():
            missing_in_bin = missing_anchors[bin_ids[missing_anchors] == bin_id]
            quota = missing_in_bin.numel()
            candidates = residual[bin_ids[residual] == bin_id]
            take = min(quota, candidates.numel())
            if take:
                order = torch.argsort(
                    flat_scores[candidates], descending=False, stable=True
                )
                displaced_parts.append(candidates[order[:take]])
            if take < quota:
                unmatched_missing_parts.append(missing_in_bin[take:])
        displaced = (
            torch.cat(displaced_parts)
            if displaced_parts
            else torch.empty(0, dtype=torch.int64)
        )
        unmatched_missing = (
            torch.cat(unmatched_missing_parts)
            if unmatched_missing_parts
            else torch.empty(0, dtype=torch.int64)
        )
        if unmatched_missing.numel():
            is_displaced = torch.zeros(triples.shape[0], dtype=torch.bool)
            is_displaced[displaced] = True
            history_displaced_parts: list[torch.Tensor] = []
            for history in torch.unique(
                histories[unmatched_missing], sorted=True
            ).tolist():
                quota = int((histories[unmatched_missing] == history).sum())
                candidates = residual[
                    (histories[residual] == history) & ~is_displaced[residual]
                ]
                take = min(quota, candidates.numel())
                if take:
                    order = torch.argsort(
                        flat_scores[candidates], descending=False, stable=True
                    )
                    history_displaced_parts.append(candidates[order[:take]])
            if history_displaced_parts:
                displaced = torch.cat([displaced, *history_displaced_parts])
        deficit = missing_anchors.numel() - displaced.numel()
        if deficit:
            is_displaced = torch.zeros(triples.shape[0], dtype=torch.bool)
            is_displaced[displaced] = True
            candidates = residual[~is_displaced[residual]]
            order = torch.argsort(
                flat_scores[candidates], descending=False, stable=True
            )
            displaced = torch.cat([displaced, candidates[order[:deficit]]])
        is_displaced = torch.zeros(triples.shape[0], dtype=torch.bool)
        is_displaced[displaced] = True
        residual = residual[~is_displaced[residual]]

    score_replacement_total = max(
        0, replacement_total - missing_anchors.numel()
    )
    if score_replacement_total == 0:
        selected = torch.cat([anchors, residual])
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return selected[torch.randperm(selected.numel(), generator=generator)]

    unique_bins, inverse, counts = torch.unique(
        bin_ids[residual], sorted=True, return_inverse=True, return_counts=True
    )
    exact = counts.to(torch.float64) * score_replacement_total / residual.numel()
    quotas = torch.floor(exact).to(torch.int64)
    leftover = score_replacement_total - int(quotas.sum())
    if leftover:
        order = torch.argsort(exact - quotas, descending=True, stable=True)
        quotas[order[:leftover]] += 1

    challenger_capacities = torch.tensor(
        [
            int(((bin_ids == bin_id) & ~is_current & ~is_anchor).sum())
            for bin_id in unique_bins.tolist()
        ],
        dtype=torch.int64,
    )
    capacities = torch.minimum(challenger_capacities, counts)
    shortfall = int(torch.clamp(quotas - capacities, min=0).sum())
    quotas = torch.minimum(quotas, capacities)
    if shortfall:
        spare = capacities - quotas
        order = torch.argsort(spare, descending=True, stable=True)
        for local_bin in order.tolist():
            take = min(shortfall, int(spare[local_bin]))
            quotas[local_bin] += take
            shortfall -= take
            if shortfall == 0:
                break
        if shortfall:
            raise ValueError("candidate pool has too few unseen challengers")

    all_indices = torch.arange(triples.shape[0], dtype=torch.int64)
    swap_plans: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for local_bin, bin_id in enumerate(unique_bins.tolist()):
        current_members = residual[inverse == local_bin]
        quota = int(quotas[local_bin])
        if not quota:
            continue
        challengers = all_indices[
            (bin_ids == bin_id) & ~is_current & ~is_anchor
        ]
        pair_count = min(quota, current_members.numel(), challengers.numel())
        if not pair_count:
            continue
        weak_current = current_members[
            torch.argsort(flat_scores[current_members], stable=True)
        ][:pair_count]
        local_selected = user_balanced_top_b_candidate_indices(
            triples[challengers], flat_scores[challengers], pair_count
        )
        strong_challengers = challengers[local_selected]
        strong_challengers = strong_challengers[
            torch.argsort(
                flat_scores[strong_challengers], descending=True, stable=True
            )
        ]
        margins = (
            flat_scores[strong_challengers] - flat_scores[weak_current]
        )
        swap_plans.append((weak_current, strong_challengers, margins))

    all_margins = (
        torch.cat([margins for _, _, margins in swap_plans])
        if swap_plans
        else torch.empty(0, dtype=flat_scores.dtype)
    )
    margin_scale = (
        float(all_margins.std(unbiased=False))
        if all_margins.numel()
        else 0.0
    )
    margin_threshold = max(
        0.0, min_standardized_swap_margin * margin_scale
    )
    displaced_parts = []
    added_parts: list[torch.Tensor] = []
    for weak_current, strong_challengers, margins in swap_plans:
        accepted = margins > margin_threshold
        displaced_parts.append(weak_current[accepted])
        added_parts.append(strong_challengers[accepted])

    displaced = (
        torch.cat(displaced_parts)
        if displaced_parts
        else torch.empty(0, dtype=torch.int64)
    )
    added = (
        torch.cat(added_parts)
        if added_parts
        else torch.empty(0, dtype=torch.int64)
    )
    is_displaced = torch.zeros(triples.shape[0], dtype=torch.bool)
    is_displaced[displaced] = True
    kept = residual[~is_displaced[residual]]
    selected = torch.cat([anchors, kept, added])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return selected[torch.randperm(selected.numel(), generator=generator)]


@dataclass(frozen=True)
class LossBandReplacementSelections:
    """Matched selections that share removals and differ only by loss band."""

    removed_indices: torch.Tensor
    added_indices: Mapping[str, torch.Tensor]
    selected_indices: Mapping[str, torch.Tensor]
    replacement_count: int


def _capacity_constrained_proportional_quotas(
    counts: torch.Tensor,
    capacities: torch.Tensor,
    total: int,
) -> torch.Tensor:
    """Allocate an exact total proportionally while respecting capacities."""
    counts = torch.as_tensor(counts, dtype=torch.int64).flatten().cpu()
    capacities = torch.as_tensor(capacities, dtype=torch.int64).flatten().cpu()
    if counts.shape != capacities.shape:
        raise ValueError("counts and capacities must have equal shape")
    if total < 0 or total > int(capacities.sum()):
        raise ValueError("requested total exceeds shared band capacity")
    quotas = torch.zeros_like(counts)
    remaining = total
    while remaining:
        spare = capacities - quotas
        active = spare > 0
        weights = counts.to(torch.float64) * active
        if not bool(active.any()):
            raise ValueError("shared band capacity was exhausted")
        if float(weights.sum()) == 0:
            weights = active.to(torch.float64)
        exact = weights * remaining / weights.sum()
        increment = torch.minimum(torch.floor(exact).to(torch.int64), spare)
        if int(increment.sum()) == 0:
            priority = exact - torch.floor(exact)
            priority[~active] = -1
            increment[int(torch.argmax(priority))] = 1
        quotas += increment
        remaining -= int(increment.sum())
    return quotas


def matched_loss_band_replacement_candidate_indices(
    candidate_triples: torch.Tensor,
    loss_scores: torch.Tensor,
    current_indices: torch.Tensor | Sequence[int],
    *,
    replacement_fraction: float,
    band_count: int = 4,
    candidate_priority_scores: Optional[torch.Tensor] = None,
    seed: int,
) -> LossBandReplacementSelections:
    """Build loss-band controls with identical removals and structure quotas.

    Unseen candidates are ranked by loss independently inside every structural
    bin and split into equal-frequency bands.  Every band and the ``loss_agnostic``
    control use the same per-bin replacement quotas and exact removed indices.
    Candidate priorities affect only selection *within* an eligible band.
    """
    triples = torch.as_tensor(candidate_triples, dtype=torch.int64).cpu()
    losses = torch.as_tensor(loss_scores).flatten().cpu()
    current = torch.as_tensor(current_indices, dtype=torch.int64).flatten().cpu()
    priorities = (
        losses
        if candidate_priority_scores is None
        else torch.as_tensor(candidate_priority_scores).flatten().cpu()
    )
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("candidate_triples must have shape [N, 3]")
    if losses.numel() != triples.shape[0] or priorities.numel() != triples.shape[0]:
        raise ValueError("scores must contain one value per candidate")
    if not torch.isfinite(losses).all() or not torch.isfinite(priorities).all():
        raise ValueError("scores must all be finite")
    if band_count < 2:
        raise ValueError("band_count must be at least two")
    if not 0 <= replacement_fraction <= 1:
        raise ValueError("replacement_fraction must be between zero and one")
    if torch.unique(current).numel() != current.numel():
        raise ValueError("current indices must be unique")
    if current.numel() and (int(current.min()) < 0 or int(current.max()) >= triples.shape[0]):
        raise ValueError("current index is outside the candidate pool")

    num_users = int(torch.unique(triples[:, 0]).numel())
    anchors = latest_coverage_candidate_indices(triples, num_users, seed=seed)
    is_current = torch.zeros(triples.shape[0], dtype=torch.bool)
    is_current[current] = True
    if not bool(is_current[anchors].all()):
        raise ValueError("current selection must contain every latest-coverage anchor")
    is_anchor = torch.zeros(triples.shape[0], dtype=torch.bool)
    is_anchor[anchors] = True
    residual = current[~is_anchor[current]]
    unseen = torch.arange(triples.shape[0], dtype=torch.int64)[~is_current & ~is_anchor]

    histories = triples[:, 2] - triples[:, 1]
    target_bins = torch.floor(torch.log2((triples[:, 2] + 1).to(torch.float64))).to(torch.int64)
    bin_ids = histories * (int(target_bins.max()) + 1) + target_bins
    unique_bins = torch.unique(bin_ids[residual], sorted=True)
    current_counts = torch.tensor(
        [int((bin_ids[residual] == value).sum()) for value in unique_bins.tolist()],
        dtype=torch.int64,
    )
    shared_capacities = torch.tensor(
        [int((bin_ids[unseen] == value).sum()) // band_count for value in unique_bins.tolist()],
        dtype=torch.int64,
    )
    replacement_total = round(residual.numel() * replacement_fraction)
    quotas = _capacity_constrained_proportional_quotas(
        current_counts, shared_capacities, replacement_total
    )

    removal_generator = torch.Generator(device="cpu").manual_seed(seed)
    removed_parts: list[torch.Tensor] = []
    eligible_by_label: dict[str, list[torch.Tensor]] = {
        f"loss_q{100 * band // band_count:02d}_{100 * (band + 1) // band_count:02d}": []
        for band in range(band_count)
    }
    eligible_by_label["loss_agnostic"] = []
    quotas_by_part: list[int] = []
    for position, bin_value in enumerate(unique_bins.tolist()):
        quota = int(quotas[position])
        if not quota:
            continue
        current_bin = residual[bin_ids[residual] == bin_value]
        permutation = torch.randperm(current_bin.numel(), generator=removal_generator)
        removed_parts.append(current_bin[permutation[:quota]])
        unseen_bin = unseen[bin_ids[unseen] == bin_value]
        ordered = unseen_bin[torch.argsort(losses[unseen_bin], stable=True)]
        for band, part in enumerate(torch.tensor_split(ordered, band_count)):
            label = f"loss_q{100 * band // band_count:02d}_{100 * (band + 1) // band_count:02d}"
            eligible_by_label[label].append(part)
        eligible_by_label["loss_agnostic"].append(unseen_bin)
        quotas_by_part.append(quota)

    removed = torch.cat(removed_parts) if removed_parts else torch.empty(0, dtype=torch.int64)
    removed_mask = torch.zeros(triples.shape[0], dtype=torch.bool)
    removed_mask[removed] = True
    kept = current[~removed_mask[current]]
    added_by_label: dict[str, torch.Tensor] = {}
    selected_by_label: dict[str, torch.Tensor] = {}
    for label, eligible_parts in eligible_by_label.items():
        chosen_parts: list[torch.Tensor] = []
        for eligible, quota in zip(eligible_parts, quotas_by_part):
            local = user_balanced_top_b_candidate_indices(
                triples[eligible], priorities[eligible], quota
            )
            chosen_parts.append(eligible[local])
        added = torch.cat(chosen_parts) if chosen_parts else torch.empty(0, dtype=torch.int64)
        selected = torch.cat([kept, added])
        shuffle_generator = torch.Generator(device="cpu").manual_seed(seed)
        selected = selected[torch.randperm(selected.numel(), generator=shuffle_generator)]
        added_by_label[label] = added
        selected_by_label[label] = selected

    return LossBandReplacementSelections(
        removed_indices=removed,
        added_indices=added_by_label,
        selected_indices=selected_by_label,
        replacement_count=replacement_total,
    )


def top_b_candidate_indices(scores: torch.Tensor, budget: int) -> torch.Tensor:
    """Return global indices of the B largest finite scalar scores."""
    flat_scores = torch.as_tensor(scores).flatten().cpu()
    _validate_budget(flat_scores.numel(), budget)
    if not torch.isfinite(flat_scores).all():
        raise ValueError("scores must all be finite")
    return torch.argsort(flat_scores, descending=True, stable=True)[:budget]


def save_selected_candidate_indices(
    path: str | Path,
    selected_indices: torch.Tensor | Sequence[int],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = torch.as_tensor(selected_indices, dtype=torch.int64).flatten()
    torch.save(
        {
            "format_version": SELECTION_FORMAT_VERSION,
            "selected_candidate_indices": indices,
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_selected_candidate_indices(path: str | Path) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload.to(dtype=torch.int64).flatten()
    if not isinstance(payload, dict):
        raise ValueError("selection file must contain a tensor or mapping")
    for key in ("selected_candidate_indices", "selected_indices"):
        if key in payload:
            return torch.as_tensor(payload[key], dtype=torch.int64).flatten()
    raise ValueError("selection file does not contain candidate indices")


@dataclass
class SubsequenceCandidateStore:
    """Compact user sequences plus unique effective-subsequence triples."""

    tokens: torch.Tensor
    offsets: torch.Tensor
    user_ids: torch.Tensor
    candidate_triples: torch.Tensor
    sid_hierarchy: int
    sequence_length: int
    metadata: Mapping[str, Any]

    @property
    def num_rows(self) -> int:
        return int(self.offsets.numel() - 1)

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_triples.shape[0])

    @property
    def max_items(self) -> int:
        return self.sequence_length // self.sid_hierarchy

    @classmethod
    def from_sequences(
        cls,
        sequences: Iterable[torch.Tensor | Sequence[int]],
        user_ids: Iterable[int],
        *,
        sid_hierarchy: int,
        sequence_length: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "SubsequenceCandidateStore":
        if sid_hierarchy <= 0:
            raise ValueError("sid_hierarchy must be positive")
        if sequence_length < 2 * sid_hierarchy:
            raise ValueError("sequence_length must fit a history item and a target")
        if sequence_length % sid_hierarchy:
            raise ValueError("sequence_length must be divisible by sid_hierarchy")

        sequence_list = [_as_flat_int_tensor(sequence) for sequence in sequences]
        user_id_list = [int(user_id) for user_id in user_ids]
        if len(sequence_list) != len(user_id_list):
            raise ValueError("sequences and user_ids must have the same length")

        max_items = sequence_length // sid_hierarchy
        offsets = [0]
        candidate_triples = []
        for user_row, sequence in enumerate(sequence_list):
            if sequence.numel() % sid_hierarchy:
                raise ValueError(
                    f"row {user_row} has {sequence.numel()} tokens, which is not "
                    f"divisible by sid_hierarchy={sid_hierarchy}"
                )
            item_count = sequence.numel() // sid_hierarchy
            offsets.append(offsets[-1] + sequence.numel())
            for target_position in range(1, item_count):
                minimum_effective_start = max(
                    0, target_position + 1 - max_items
                )
                candidate_triples.extend(
                    (user_row, effective_start, target_position)
                    for effective_start in range(
                        minimum_effective_start, target_position
                    )
                )

        tokens = (
            torch.cat(sequence_list)
            if sequence_list
            else torch.empty(0, dtype=torch.int32)
        )
        triples = torch.tensor(candidate_triples, dtype=torch.int32).reshape(-1, 3)
        store = cls(
            tokens=tokens,
            offsets=torch.tensor(offsets, dtype=torch.int64),
            user_ids=torch.tensor(user_id_list, dtype=torch.int64),
            candidate_triples=triples,
            sid_hierarchy=sid_hierarchy,
            sequence_length=sequence_length,
            metadata=dict(metadata or {}),
        )
        store.validate()
        return store

    def validate(self) -> None:
        if self.sid_hierarchy <= 0:
            raise ValueError("sid_hierarchy must be positive")
        if self.sequence_length % self.sid_hierarchy:
            raise ValueError("sequence_length must be divisible by sid_hierarchy")
        if self.max_items < 2:
            raise ValueError("sequence_length must fit history plus target")
        if self.offsets.ndim != 1 or self.offsets.numel() == 0:
            raise ValueError("offsets must be a non-empty one-dimensional tensor")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != self.tokens.numel():
            raise ValueError("offsets must span the complete token tensor")
        if torch.any(self.offsets[1:] < self.offsets[:-1]):
            raise ValueError("offsets must be monotonic")
        if torch.any((self.offsets[1:] - self.offsets[:-1]) % self.sid_hierarchy):
            raise ValueError("row token counts must align to sid_hierarchy")
        if self.user_ids.numel() != self.num_rows:
            raise ValueError("user_ids must contain one value per user row")
        if self.candidate_triples.ndim != 2 or self.candidate_triples.shape[1] != 3:
            raise ValueError("candidate_triples must have shape [N, 3]")
        if not self.num_candidates:
            return

        triples = self.candidate_triples.to(torch.int64)
        user_rows, starts, targets = triples.unbind(dim=1)
        if int(user_rows.min()) < 0 or int(user_rows.max()) >= self.num_rows:
            raise ValueError("candidate contains an invalid user_row")
        item_counts = (
            (self.offsets[1:] - self.offsets[:-1]) // self.sid_hierarchy
        )
        if torch.any(starts < 0) or torch.any(starts >= targets):
            raise ValueError("candidate must contain history before its target")
        if torch.any(targets >= item_counts[user_rows]):
            raise ValueError("candidate target is outside its user row")
        if torch.any(targets + 1 - starts > self.max_items):
            raise ValueError("candidate exceeds the effective sequence length")
        if torch.unique(triples, dim=0).shape[0] != self.num_candidates:
            raise ValueError("candidate_triples contains duplicates")

    def length_histogram(self) -> dict[int, int]:
        if not self.num_candidates:
            return {}
        lengths = (
            self.candidate_triples[:, 2].to(torch.int64)
            - self.candidate_triples[:, 1].to(torch.int64)
            + 1
        )
        counts = torch.bincount(lengths, minlength=self.max_items + 1)
        return {
            length: int(counts[length])
            for length in range(2, self.max_items + 1)
            if int(counts[length])
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": FORMAT_VERSION,
                "tokens": self.tokens,
                "offsets": self.offsets,
                "user_ids": self.user_ids,
                "candidate_triples": self.candidate_triples,
                "sid_hierarchy": self.sid_hierarchy,
                "sequence_length": self.sequence_length,
                "metadata": dict(self.metadata),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SubsequenceCandidateStore":
        payload = torch.load(Path(path), map_location="cpu")
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported candidate store version: {payload.get('format_version')}"
            )
        store = cls(
            tokens=payload["tokens"],
            offsets=payload["offsets"],
            user_ids=payload["user_ids"],
            candidate_triples=payload["candidate_triples"],
            sid_hierarchy=int(payload["sid_hierarchy"]),
            sequence_length=int(payload["sequence_length"]),
            metadata=payload.get("metadata", {}),
        )
        store.validate()
        return store


def genpas_candidate_weights(
    store: SubsequenceCandidateStore,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """Return GenPAS probabilities pushed through suffix truncation/dedup."""
    for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"{name} must be finite")
    if not store.num_candidates:
        return torch.empty(0, dtype=torch.float64)

    triples = store.candidate_triples.to(dtype=torch.int64, device="cpu")
    user_rows, starts, targets = triples.unbind(dim=1)
    item_counts = (
        (store.offsets[1:] - store.offsets[:-1]) // store.sid_hierarchy
    ).to(dtype=torch.int64, device="cpu")
    valid_target_counts = item_counts - 1

    user_mass = valid_target_counts.to(torch.float64).pow(alpha)
    target_normalizers = torch.empty(store.num_rows, dtype=torch.float64)
    for user_row, target_count in enumerate(valid_target_counts.tolist()):
        positions = torch.arange(1, target_count + 1, dtype=torch.float64)
        target_normalizers[user_row] = positions.pow(beta).sum()
    target_mass = targets.to(torch.float64).pow(beta) / target_normalizers[user_rows]

    max_target = int(targets.max())
    input_normalizers = torch.empty(max_target + 1, dtype=torch.float64)
    input_normalizers[0] = torch.nan
    for target in range(1, max_target + 1):
        positions = torch.arange(1, target + 1, dtype=torch.float64)
        input_normalizers[target] = positions.pow(gamma).sum()

    minimum_starts = torch.clamp(targets + 1 - store.max_items, min=0)
    start_mass = (starts + 1).to(torch.float64).pow(gamma)
    collapsed = starts == minimum_starts
    if collapsed.any():
        cumulative_mass = torch.empty(max_target, dtype=torch.float64)
        for start in range(max_target):
            positions = torch.arange(1, start + 2, dtype=torch.float64)
            cumulative_mass[start] = positions.pow(gamma).sum()
        start_mass[collapsed] = cumulative_mass[starts[collapsed]]
    input_mass = start_mass / input_normalizers[targets]

    weights = user_mass[user_rows] * target_mass * input_mass
    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError("GenPAS produced non-positive or non-finite weights")
    return weights / weights.sum()


class SubsequenceCandidateDataset(Dataset):
    """Map-style examples selected from the fixed global candidate pool."""

    def __init__(
        self,
        store: SubsequenceCandidateStore | str | Path,
        selected_indices: Optional[torch.Tensor | Sequence[int]] = None,
    ) -> None:
        self.store = (
            SubsequenceCandidateStore.load(store)
            if isinstance(store, (str, Path))
            else store
        )
        self.store.validate()
        if selected_indices is None:
            self.selected_indices = torch.arange(
                self.store.num_candidates, dtype=torch.int64
            )
        else:
            self.selected_indices = torch.as_tensor(
                selected_indices, dtype=torch.int64
            ).flatten()
            if self.selected_indices.numel() and (
                int(self.selected_indices.min()) < 0
                or int(self.selected_indices.max()) >= self.store.num_candidates
            ):
                raise IndexError("selected candidate index is outside the store")
            if torch.unique(self.selected_indices).numel() != self.selected_indices.numel():
                raise ValueError("selected candidate indices must be unique")

    def __len__(self) -> int:
        return int(self.selected_indices.numel())

    def candidate_key(self, index: int) -> tuple[int, int, int]:
        candidate_index = int(self.selected_indices[index])
        key = self.store.candidate_triples[candidate_index].tolist()
        return int(key[0]), int(key[1]), int(key[2])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_row, effective_start, target_position = self.candidate_key(index)
        hierarchy = self.store.sid_hierarchy
        row_start = int(self.store.offsets[user_row])
        row_end = int(self.store.offsets[user_row + 1])
        row_tokens = self.store.tokens[row_start:row_end]
        sequence = row_tokens[
            effective_start * hierarchy : (target_position + 1) * hierarchy
        ]
        return {
            "sequence_data": sequence.clone(),
            "user_id": self.store.user_ids[user_row].reshape(1).clone(),
        }
