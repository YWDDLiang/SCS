from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from src.data.selection.subsequence_candidate import latest_coverage_candidate_indices


@dataclass(frozen=True)
class RefinementProbeDecision:
    action: str
    reason: str
    current_margin_mass: float
    reference_margin_mass: float
    current_runway_updates: float
    reference_runway_updates: float
    opportunity_ratio: float
    score_correlation: float
    normalized_score_drift: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def deterministic_probe_indices(
    candidate_count: int, *, fraction: float, seed: int
) -> torch.Tensor:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be within (0, 1]")
    indices = torch.arange(candidate_count, dtype=torch.int64)
    values = (indices * 1103515245 + seed * 1013904223) & 0x7FFFFFFF
    return indices[values < int(fraction * (1 << 31))]


def sampled_stagewise_swap_margins(
    triples: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_scores: torch.Tensor,
    current: torch.Tensor,
    *,
    replacement_fraction_cap: float,
    anchor_seed: int,
) -> torch.Tensor:
    if not 0 <= replacement_fraction_cap <= 1:
        raise ValueError("replacement_fraction_cap must be within [0, 1]")
    sample_indices = torch.as_tensor(sample_indices, dtype=torch.int64).flatten()
    sample_scores = torch.as_tensor(sample_scores, dtype=torch.float32).flatten()
    current = torch.as_tensor(current, dtype=torch.int64).flatten()
    if sample_indices.numel() != sample_scores.numel():
        raise ValueError("sample indices and scores must have equal length")
    if torch.unique(sample_indices).numel() != sample_indices.numel():
        raise ValueError("sample indices must be unique")
    score_by_index = torch.full((triples.shape[0],), torch.nan, dtype=torch.float32)
    score_by_index[sample_indices] = sample_scores
    sampled = torch.zeros(triples.shape[0], dtype=torch.bool)
    sampled[sample_indices] = True
    is_current = torch.zeros(triples.shape[0], dtype=torch.bool)
    is_current[current] = True
    anchors = latest_coverage_candidate_indices(
        triples, int(torch.unique(triples[:, 0]).numel()), seed=anchor_seed
    )
    is_anchor = torch.zeros(triples.shape[0], dtype=torch.bool)
    is_anchor[anchors] = True
    residual = current[~is_anchor[current] & sampled[current]]
    challengers = sample_indices[~is_current[sample_indices] & ~is_anchor[sample_indices]]
    histories = triples[:, 2] - triples[:, 1]
    target_bins = torch.floor(
        torch.log2((triples[:, 2] + 1).to(torch.float64))
    ).to(torch.int64)
    bin_ids = histories * (int(target_bins.max()) + 1) + target_bins
    chunks: list[torch.Tensor] = []
    for bin_id in torch.unique(bin_ids[residual], sorted=True).tolist():
        current_bin = residual[bin_ids[residual] == bin_id]
        challenger_bin = challengers[bin_ids[challengers] == bin_id]
        pair_count = min(
            current_bin.numel(),
            challenger_bin.numel(),
            round(current_bin.numel() * replacement_fraction_cap),
        )
        if not pair_count:
            continue
        weak = current_bin[
            torch.argsort(score_by_index[current_bin], stable=True)
        ]
        strong = challenger_bin[
            torch.argsort(
                score_by_index[challenger_bin], descending=True, stable=True
            )
        ]
        chunks.append(
            score_by_index[strong[:pair_count]] - score_by_index[weak[:pair_count]]
        )
    return torch.cat(chunks) if chunks else torch.empty(0)


def positive_margin_mass_per_pair(margins: torch.Tensor) -> float:
    margins = torch.as_tensor(margins, dtype=torch.float32).flatten()
    if not margins.numel():
        return 0.0
    return float(torch.clamp_min(margins, 0).sum() / margins.numel())


def score_drift(
    previous_scores: torch.Tensor, current_scores: torch.Tensor
) -> tuple[float, float]:
    previous = torch.as_tensor(previous_scores, dtype=torch.float32).flatten()
    current = torch.as_tensor(current_scores, dtype=torch.float32).flatten()
    if previous.numel() != current.numel() or previous.numel() < 2:
        raise ValueError("score snapshots must have the same length of at least two")
    previous_centered = previous - previous.mean()
    current_centered = current - current.mean()
    denominator = previous_centered.std(unbiased=False) * current_centered.std(
        unbiased=False
    )
    correlation = float(
        (previous_centered * current_centered).mean() / max(float(denominator), 1e-12)
    )
    correlation = max(-1.0, min(1.0, correlation))
    normalized_drift = float(
        (current - previous).std(unbiased=False)
        / max(float(previous.std(unbiased=False)), 1e-12)
    )
    return correlation, normalized_drift


def decide_refinement_probe(
    *,
    current_margins: torch.Tensor,
    reference_margins: torch.Tensor,
    current_runway_updates: float,
    reference_runway_updates: float,
    previous_scores: torch.Tensor,
    current_scores: torch.Tensor,
    min_relative_opportunity: float = 0.5,
) -> RefinementProbeDecision:
    if current_runway_updates < 0 or reference_runway_updates <= 0:
        raise ValueError("runway updates must be non-negative with positive reference")
    if min_relative_opportunity < 0:
        raise ValueError("min_relative_opportunity must be non-negative")
    current_mass = positive_margin_mass_per_pair(current_margins)
    reference_mass = positive_margin_mass_per_pair(reference_margins)
    reference_opportunity = reference_mass * reference_runway_updates
    current_opportunity = current_mass * current_runway_updates
    ratio = (
        current_opportunity / reference_opportunity
        if reference_opportunity > 0
        else 0.0
    )
    correlation, normalized_drift = score_drift(previous_scores, current_scores)
    accepted = ratio >= min_relative_opportunity
    return RefinementProbeDecision(
        action="full_score" if accepted else "skip_refinement",
        reason=(
            "sampled_opportunity_is_compute_worthy"
            if accepted
            else "sampled_opportunity_below_reference"
        ),
        current_margin_mass=current_mass,
        reference_margin_mass=reference_mass,
        current_runway_updates=current_runway_updates,
        reference_runway_updates=reference_runway_updates,
        opportunity_ratio=ratio,
        score_correlation=correlation,
        normalized_score_drift=normalized_drift,
    )
