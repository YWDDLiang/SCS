"""Pure rules for coverage-aware spectrum event prioritization."""

from __future__ import annotations

import numpy as np


def augmentation_eligibility(rows: np.ndarray) -> np.ndarray:
    """Identify events whose histories can be changed without altering length."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError("rows must have shape [N, 3] as (user, start, target)")
    history_lengths = rows[:, 2] - rows[:, 1]
    return (history_lengths >= 2) & (rows[:, 2] > history_lengths)


def cross_user_transition_coverage(store) -> np.ndarray:
    """Count distinct raw-TRAIN users sharing each adjacent item transition.

    The returned table is indexed by ``[user_row, target_position]``. Each
    item is represented by its complete SID vector, and repeated transitions
    by one user count once.
    """
    hierarchy = int(store.sid_hierarchy)
    raw_tokens = store.tokens.cpu() if hasattr(store.tokens, "cpu") else store.tokens
    raw_offsets = store.offsets.cpu() if hasattr(store.offsets, "cpu") else store.offsets
    tokens = np.asarray(raw_tokens, dtype=np.int32)
    offsets = np.asarray(raw_offsets, dtype=np.int64)
    if hierarchy <= 0 or len(offsets) != len(store.user_ids) + 1:
        raise ValueError("invalid raw TRAIN candidate store")

    features = []
    user_parts = []
    target_parts = []
    maximum_target = 0
    for user_row in range(len(offsets) - 1):
        user_items = tokens[offsets[user_row]:offsets[user_row + 1]].reshape(-1, hierarchy)
        transitions = len(user_items) - 1
        if transitions <= 0:
            continue
        features.append(np.concatenate((user_items[:-1], user_items[1:]), axis=1))
        user_parts.append(np.full(transitions, user_row, dtype=np.int32))
        target_parts.append(np.arange(1, transitions + 1, dtype=np.int32))
        maximum_target = max(maximum_target, transitions)

    coverage_table = np.zeros((len(offsets) - 1, maximum_target + 1), dtype=np.int32)
    if not features:
        return coverage_table
    edge_features = np.concatenate(features, axis=0)
    transition_users = np.concatenate(user_parts)
    transition_targets = np.concatenate(target_parts)
    _, transition_edge_ids = np.unique(edge_features, axis=0, return_inverse=True)
    user_edge_pairs = np.column_stack((transition_users, transition_edge_ids))
    distinct_user_edges = np.unique(user_edge_pairs, axis=0)
    coverage_by_edge = np.bincount(
        distinct_user_edges[:, 1], minlength=int(transition_edge_ids.max()) + 1
    ).astype(np.int32)
    coverage_table[transition_users, transition_targets] = coverage_by_edge[transition_edge_ids]
    if np.any(coverage_table[transition_users, transition_targets] <= 0):
        raise RuntimeError("raw TRAIN transition coverage must be positive")
    return coverage_table


def coverage_for_events(
    rows: np.ndarray,
    store,
    coverage_by_user_target: np.ndarray,
    *,
    sorted_user_ids: np.ndarray | None = None,
    user_rows_in_sorted_order: np.ndarray | None = None,
) -> np.ndarray:
    """Look up event coverage using source user IDs, not candidate row IDs."""
    rows = np.asarray(rows, dtype=np.int64)
    if sorted_user_ids is None or user_rows_in_sorted_order is None:
        user_ids = store.user_ids.cpu().numpy() if hasattr(store.user_ids, "cpu") else np.asarray(store.user_ids)
        order = np.argsort(user_ids, kind="stable")
        sorted_ids = user_ids[order]
    else:
        sorted_ids = np.asarray(sorted_user_ids)
        order = np.asarray(user_rows_in_sorted_order, dtype=np.int64)
        if sorted_ids.shape != order.shape:
            raise ValueError("sorted user IDs and row mapping must align")
    locations = np.searchsorted(sorted_ids, rows[:, 0])
    if np.any(locations >= len(sorted_ids)) or not np.array_equal(sorted_ids[locations], rows[:, 0]):
        raise KeyError("raw event contains a user ID absent from the original TRAIN store")
    user_rows = order[locations]
    targets = rows[:, 2]
    if np.any(targets < 0) or np.any(targets >= coverage_by_user_target.shape[1]):
        raise IndexError("raw event target position is outside the coverage table")
    coverage = coverage_by_user_target[user_rows, targets]
    if np.any(coverage <= 0):
        raise RuntimeError("raw event transition coverage must be positive")
    return coverage


def coverage_aware_spectrum_probabilities(
    spectrum_probabilities: np.ndarray,
    transition_coverage: np.ndarray,
    view_ready: np.ndarray,
    *,
    minimum_coverage: int = 2,
    view_readiness_weight: float = 4.0,
) -> np.ndarray:
    """Combine spectrum scores and transition coverage into one priority.

    Additional presentations are sampled only from events meeting the
    coverage criterion. Among those candidates, events that admit an
    alternative history receive the configured priority multiplier.
    """
    spectrum_probabilities = np.asarray(spectrum_probabilities, dtype=np.float64)
    transition_coverage = np.asarray(transition_coverage, dtype=np.int64)
    view_ready = np.asarray(view_ready, dtype=bool)
    minimum_coverage = int(minimum_coverage)
    view_readiness_weight = float(view_readiness_weight)
    if (
        spectrum_probabilities.ndim != 1
        or transition_coverage.shape != spectrum_probabilities.shape
        or view_ready.shape != spectrum_probabilities.shape
    ):
        raise ValueError("spectrum scores, coverage, and view readiness must align")
    if not np.isfinite(spectrum_probabilities).all() or np.any(spectrum_probabilities < 0):
        raise ValueError("spectrum probabilities must be finite and non-negative")
    if spectrum_probabilities.sum() <= 0:
        raise ValueError("spectrum probabilities must have positive mass")
    if minimum_coverage < 1:
        raise ValueError("minimum coverage must be positive")
    if not np.isfinite(view_readiness_weight) or view_readiness_weight <= 0:
        raise ValueError("view candidate weight must be finite and positive")

    coverage_qualified = transition_coverage >= minimum_coverage
    weighted = spectrum_probabilities * np.where(view_ready, view_readiness_weight, 1.0)
    weighted *= coverage_qualified
    total = weighted.sum()
    if total > 0:
        weighted /= total
    return weighted


def partition_integer_budget(total: int, world_size: int) -> np.ndarray:
    """Split a global integer quota across ranks without changing its sum."""
    total = int(total)
    world_size = int(world_size)
    if total < 0 or world_size <= 0:
        raise ValueError("total must be non-negative and world_size positive")
    base, remainder = divmod(total, world_size)
    budgets = np.full(world_size, base, dtype=np.int64)
    budgets[:remainder] += 1
    if int(budgets.sum()) != total:
        raise RuntimeError("rank budget partition changed the global total")
    return budgets


def sample_scheduled_presentations(
    probabilities: np.ndarray,
    augmentation_probabilities: np.ndarray,
    view_eligible: np.ndarray,
    extra_count: int,
    augmentation_count: int,
    rng: np.random.Generator,
    candidate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample fixed repeat/view quotas; views are restricted to gated rows.

    Spectrum probabilities select both action pools. Conflict probabilities
    only reweight the eligible view pool. Sampling is with replacement, as in
    the original fixed-exposure scheduler.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    action_probabilities = np.asarray(augmentation_probabilities, dtype=np.float64)
    eligible = np.asarray(view_eligible, dtype=bool)
    allowed = (
        np.ones_like(eligible, dtype=bool)
        if candidate_mask is None
        else np.asarray(candidate_mask, dtype=bool)
    )
    if (
        probabilities.ndim != 1
        or probabilities.shape != action_probabilities.shape
        or probabilities.shape != eligible.shape
        or probabilities.shape != allowed.shape
    ):
        raise ValueError("event probabilities, action probabilities, and candidate mask must align")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("event probabilities must be finite, non-negative, and nonzero")
    if not np.isfinite(action_probabilities).all() or np.any(action_probabilities < 0):
        raise ValueError("action probabilities must be finite and non-negative")
    if not 0 <= augmentation_count <= extra_count:
        raise ValueError("augmentation_count must lie within the extra exposure count")
    if extra_count < 0:
        raise ValueError("extra_count must be non-negative")
    if not np.any(allowed):
        raise ValueError("candidate pool is empty")
    allowed_weights = probabilities * allowed
    allowed_total = allowed_weights.sum()
    if not np.isfinite(allowed_total) or allowed_total <= 0:
        raise ValueError("candidate pool has no positive probability mass")
    allowed_weights = allowed_weights / allowed_total

    repeats = extra_count - augmentation_count
    if augmentation_count:
        view_weights = allowed_weights * action_probabilities * eligible
        total = view_weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("fixed augmentation quota has no eligible spectrum/conflict mass")
        views = rng.choice(len(probabilities), size=augmentation_count, replace=True, p=view_weights / total)
    else:
        views = np.empty(0, dtype=np.int64)
    repeated = (
        rng.choice(len(probabilities), size=repeats, replace=True, p=allowed_weights)
        if repeats
        else np.empty(0, dtype=np.int64)
    )
    selected = np.concatenate((views, repeated)).astype(np.int64, copy=False)
    actions = np.concatenate((np.ones(len(views), dtype=bool), np.zeros(len(repeated), dtype=bool)))
    order = rng.permutation(len(selected))
    return selected[order], actions[order]
