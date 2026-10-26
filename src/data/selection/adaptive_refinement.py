from __future__ import annotations

from typing import Sequence

import torch


def _flat_indices(
    values: torch.Tensor | Sequence[int],
    *,
    upper_bound: int,
    name: str,
) -> torch.Tensor:
    indices = torch.as_tensor(values, dtype=torch.int64).flatten().cpu()
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= upper_bound
    ):
        raise IndexError(f"{name} contains an index outside [0, {upper_bound})")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError(f"{name} must contain unique indices")
    return indices


def random_challenger_pool(
    num_candidates: int,
    current_indices: torch.Tensor | Sequence[int],
    num_challengers: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Keep the current subset and uniformly add unseen global challengers."""
    if num_candidates < 0:
        raise ValueError("num_candidates must be non-negative")
    current = _flat_indices(
        current_indices, upper_bound=num_candidates, name="current_indices"
    )
    remaining = num_candidates - current.numel()
    if not 0 <= num_challengers <= remaining:
        raise ValueError(
            f"num_challengers must be between 0 and {remaining}, got {num_challengers}"
        )
    if not num_challengers:
        return current.clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    excluded = set(current.tolist())
    challengers: list[int] = []
    challenger_set: set[int] = set()
    while len(challengers) < num_challengers:
        needed = num_challengers - len(challengers)
        draw_count = min(max(needed * 2, 64), max(num_candidates, 1))
        draws = torch.randint(
            num_candidates, (draw_count,), generator=generator
        ).tolist()
        for candidate in draws:
            if candidate in excluded or candidate in challenger_set:
                continue
            challenger_set.add(candidate)
            challengers.append(candidate)
            if len(challengers) == num_challengers:
                break
    return torch.cat(
        [current, torch.tensor(challengers, dtype=torch.int64)]
    )


def topk_membership_probability(
    predicted_scores: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    """Estimate Top-K membership from bootstrap high-fidelity predictions."""
    predictions = torch.as_tensor(predicted_scores).detach().float().cpu()
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(0)
    if predictions.ndim != 2 or predictions.shape[0] == 0:
        raise ValueError("predicted_scores must have shape [E, N] or [N]")
    if not 0 <= budget <= predictions.shape[1]:
        raise ValueError("budget must be between 0 and the candidate-pool size")
    if not torch.isfinite(predictions).all():
        raise ValueError("predicted_scores must all be finite")
    probabilities = torch.zeros(predictions.shape[1], dtype=torch.float32)
    if not budget:
        return probabilities
    for row in predictions:
        topk = torch.argsort(row, descending=True, stable=True)[:budget]
        probabilities[topk] += 1.0
    return probabilities / predictions.shape[0]


def adaptive_refinement_indices(
    candidate_indices: torch.Tensor | Sequence[int],
    predicted_scores: torch.Tensor,
    budget: int,
    max_refinements: int,
    *,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """Choose global candidates whose predicted Top-K membership is uncertain.

    With bootstrap predictions, candidates with membership probability in
    ``(epsilon, 1 - epsilon)`` are refined first. With a single prediction,
    there is no empirical uncertainty, so the method refines the candidates
    closest to the predicted Top-K boundary.
    """
    candidates = torch.as_tensor(candidate_indices, dtype=torch.int64).flatten().cpu()
    predictions = torch.as_tensor(predicted_scores).detach().float().cpu()
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(0)
    if predictions.ndim != 2 or predictions.shape[1] != candidates.numel():
        raise ValueError("predicted_scores must align with candidate_indices")
    if not 0 <= budget <= candidates.numel():
        raise ValueError("budget must be between 0 and the candidate-pool size")
    if not 0 <= max_refinements <= candidates.numel():
        raise ValueError("max_refinements must fit the candidate-pool size")
    if not 0 <= epsilon < 0.5:
        raise ValueError("epsilon must be in [0, 0.5)")
    if not max_refinements or not candidates.numel():
        return torch.empty(0, dtype=torch.int64)

    mean_scores = predictions.mean(dim=0)
    if budget == 0:
        cutoff = mean_scores.max() + 1.0
    else:
        cutoff = torch.sort(mean_scores, descending=True, stable=True).values[
            budget - 1
        ]
    boundary_distance = torch.abs(mean_scores - cutoff)

    if predictions.shape[0] == 1:
        local_order = torch.argsort(boundary_distance, stable=True)
    else:
        probabilities = topk_membership_probability(predictions, budget)
        uncertain_mask = (probabilities > epsilon) & (
            probabilities < 1.0 - epsilon
        )
        uncertain_local = torch.nonzero(uncertain_mask, as_tuple=False).flatten()
        if not uncertain_local.numel():
            return torch.empty(0, dtype=torch.int64)
        ambiguity = torch.abs(probabilities[uncertain_local] - 0.5)
        order = sorted(
            range(uncertain_local.numel()),
            key=lambda position: (
                float(ambiguity[position]),
                float(boundary_distance[uncertain_local[position]]),
                int(candidates[uncertain_local[position]]),
            ),
        )
        local_order = uncertain_local[torch.tensor(order, dtype=torch.int64)]
    return candidates[local_order[:max_refinements]]


def finalize_refined_selection(
    candidate_indices: torch.Tensor | Sequence[int],
    predicted_scores: torch.Tensor,
    precise_indices: torch.Tensor | Sequence[int],
    precise_scores: torch.Tensor | Sequence[float],
    budget: int,
) -> torch.Tensor:
    """Replace predicted scores with precise scores and return global Top-K."""
    candidates = torch.as_tensor(candidate_indices, dtype=torch.int64).flatten().cpu()
    predictions = torch.as_tensor(predicted_scores).detach().float().cpu()
    if predictions.ndim == 2:
        predictions = predictions.mean(dim=0)
    if predictions.ndim != 1 or predictions.numel() != candidates.numel():
        raise ValueError("predicted_scores must align with candidate_indices")
    if not 0 <= budget <= candidates.numel():
        raise ValueError("budget must be between 0 and the candidate-pool size")

    precise_global = torch.as_tensor(precise_indices, dtype=torch.int64).flatten().cpu()
    precise_values = torch.as_tensor(precise_scores).detach().float().flatten().cpu()
    if precise_global.numel() != precise_values.numel():
        raise ValueError("precise_indices and precise_scores must have equal length")
    if torch.unique(precise_global).numel() != precise_global.numel():
        raise ValueError("precise_indices must be unique")
    if not torch.isfinite(predictions).all() or not torch.isfinite(precise_values).all():
        raise ValueError("selection scores must all be finite")

    positions = {int(candidate): position for position, candidate in enumerate(candidates)}
    final_scores = predictions.clone()
    for candidate, score in zip(precise_global.tolist(), precise_values.tolist()):
        if candidate not in positions:
            raise ValueError(f"precise candidate {candidate} is outside the pool")
        final_scores[positions[candidate]] = score
    local_topk = torch.argsort(final_scores, descending=True, stable=True)[:budget]
    return candidates[local_topk]


def reference_gradient_drift(
    previous_gradient: torch.Tensor,
    current_gradient: torch.Tensor,
) -> float:
    """Return one minus cosine similarity between reference gradients."""
    previous = torch.as_tensor(previous_gradient).detach().float().flatten().cpu()
    current = torch.as_tensor(current_gradient).detach().float().flatten().cpu()
    if previous.shape != current.shape:
        raise ValueError("reference gradients must have the same shape")
    if not previous.numel():
        raise ValueError("reference gradients must be non-empty")
    previous_norm = torch.linalg.vector_norm(previous)
    current_norm = torch.linalg.vector_norm(current)
    if previous_norm == 0 or current_norm == 0:
        raise ValueError("reference gradients must have non-zero norm")
    cosine = torch.dot(previous, current) / (previous_norm * current_norm)
    return float(1.0 - torch.clamp(cosine, -1.0, 1.0))
