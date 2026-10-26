"""Capacity-preserving selection over repeated literal training occurrences."""
import heapq
import numpy as np


def weighted_batch(capacities, weights, size, rng):
    """Weighted sampling without replacement over the expanded multiset.

    Each equal-weight group's first exponential arrival has rate count*weight.
    Its subsequent arrivals have rates (count-1)*weight, etc. Only the first
    `size` initial group arrivals can enter the first `size` global arrivals.
    This avoids allocating one random variable for every repeated occurrence.
    """
    capacities = np.asarray(capacities, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if (capacities.shape != weights.shape or np.any(capacities < 0)
            or np.any(~np.isfinite(weights)) or np.any(weights < 0)):
        raise ValueError("Invalid capacities or weights")
    active = np.flatnonzero((capacities > 0) & (weights > 0))
    if size < 0 or size > int(capacities[active].sum()):
        raise ValueError("Selection size exceeds positive-weight population")
    if size == 0:
        return np.empty(0, dtype=np.int64)
    arrival = rng.exponential(size=len(active)) / (capacities[active]*weights[active])
    keep = min(size, len(active))
    chosen = np.argpartition(arrival, keep-1)[:keep]
    heap = [(float(arrival[i]), int(active[i]), int(capacities[active[i]]))
            for i in chosen]
    heapq.heapify(heap)
    answer = []
    for _ in range(size):
        when, group, remaining = heapq.heappop(heap)
        answer.append(group)
        remaining -= 1
        if remaining:
            next_time = when + float(rng.exponential()) / (remaining*weights[group])
            heapq.heappush(heap, (next_time, group, remaining))
    return np.asarray(answer, dtype=np.int64)


def coverage_stratum_budgets(sizes, wanted):
    """Author DEALRec equal-coverage allocation, including sparse strata."""
    sizes = np.asarray(sizes, dtype=np.int64)
    if wanted < 0 or wanted > int(sizes.sum()):
        raise ValueError("Insufficient post-prune candidates")
    order = sorted(np.flatnonzero(sizes).tolist(), key=lambda i: (sizes[i], i))
    allocations = np.zeros_like(sizes)
    remaining = int(wanted)
    for position, index in enumerate(order):
        budget = min(int(sizes[index]), remaining // (len(order)-position))
        allocations[index] = budget
        remaining -= budget
    if remaining:
        raise RuntimeError("Coverage allocation failed to use exact budget")
    return allocations


def group_hypergeometric(capacities, wanted, rng):
    """Uniform occurrence selection within a stratum, with exact group counts."""
    capacities = np.asarray(capacities, dtype=np.int64)
    if wanted < 0 or wanted > int(capacities.sum()) or np.any(capacities < 0):
        raise ValueError("Invalid hypergeometric population")
    return rng.multivariate_hypergeometric(capacities, int(wanted), method="marginals")


def choose_literal_ids(group_counts, chosen_counts, occurrence_to_group, rng):
    """Uniform identities conditional on group counts; no training deduplication."""
    group_counts = np.asarray(group_counts, dtype=np.int64)
    chosen_counts = np.asarray(chosen_counts, dtype=np.int64)
    if (group_counts.shape != chosen_counts.shape or np.any(chosen_counts < 0)
            or np.any(chosen_counts > group_counts)):
        raise ValueError("Invalid selected capacities")
    mapping = np.asarray(occurrence_to_group)
    if int(group_counts.sum()) != len(mapping):
        raise ValueError("Capacity sum differs from literal population")
    ordered = np.argsort(mapping, kind="stable")
    offsets = np.r_[0, np.cumsum(group_counts)]
    membership = np.zeros(len(mapping), dtype=np.bool_)
    for key in np.flatnonzero(chosen_counts):
        count, chosen = int(group_counts[key]), int(chosen_counts[key])
        slots = ordered[offsets[key]:offsets[key+1]]
        if chosen == count:
            membership[slots] = True
        else:
            membership[slots[rng.choice(count, size=chosen, replace=False)]] = True
    if int(membership.sum()) != int(chosen_counts.sum()):
        raise RuntimeError("Selected identity cardinality mismatch")
    actual = np.bincount(mapping[membership], minlength=len(group_counts))
    if not np.array_equal(actual, chosen_counts):
        raise RuntimeError("Identity projection changed selected capacities")
    return membership
