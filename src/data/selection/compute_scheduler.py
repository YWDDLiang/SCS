from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class ValidationPoint:
    step: int
    value: float


@dataclass(frozen=True)
class ScheduleDecision:
    action: str
    reason: str
    step: int
    recent_gain_per_update: float | None
    reference_gain_per_update: float | None
    gain_ratio: float | None
    plateau_severity: float
    replacement_fraction_cap: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _window_gain(points: Sequence[ValidationPoint]) -> float:
    update_span = points[-1].step - points[0].step
    if update_span <= 0:
        raise ValueError("validation steps must be strictly increasing")
    return (points[-1].value - points[0].value) / update_span


def decide_schedule_action(
    points: Sequence[ValidationPoint],
    *,
    current_step: int,
    total_update_budget: int,
    refinements_used: int,
    max_refinements: int = 2,
    window: int = 5,
    min_step: int = 400,
    trigger_gain_ratio: float = 0.10,
    scoring_cost_updates: int = 15,
    min_updates_after_refinement: int = 50,
    min_replacement_fraction: float = 0.05,
    max_replacement_fraction: float = 0.15,
    refinement_started_step: int | None = None,
    pre_refinement_best: float | None = None,
    probe_updates: int = 20,
    min_relative_acceptance_gain: float = 0.002,
) -> ScheduleDecision:
    """Choose whether to train, refine the data, or stop.

    The signal is scale-free: recent validation gain per optimizer update is
    compared with the best earlier gain observed through the same-size window.
    A refinement is only allowed when enough total-compute runway remains to
    pay for scoring and for a meaningful post-selection training phase. A
    newly refined dataset must also beat the previous stage's best validation
    value during a short probe; otherwise the controller is told to roll back.
    """
    if window < 2:
        raise ValueError("window must be at least two")
    if total_update_budget < 0 or current_step < 0:
        raise ValueError("update counts must be non-negative")
    if not 0 <= refinements_used <= max_refinements:
        raise ValueError("refinements_used must fit max_refinements")
    if not 0 <= trigger_gain_ratio <= 1:
        raise ValueError("trigger_gain_ratio must be between zero and one")
    if not 0 <= min_replacement_fraction <= max_replacement_fraction <= 1:
        raise ValueError("replacement fractions must be ordered within [0, 1]")
    if probe_updates < 0:
        raise ValueError("probe_updates must be non-negative")
    if min_relative_acceptance_gain < 0:
        raise ValueError("min_relative_acceptance_gain must be non-negative")
    probe_enabled = (
        refinement_started_step is not None or pre_refinement_best is not None
    )
    if probe_enabled and (
        refinement_started_step is None or pre_refinement_best is None
    ):
        raise ValueError(
            "refinement_started_step and pre_refinement_best must be provided together"
        )
    if refinement_started_step is not None and refinement_started_step < 0:
        raise ValueError("refinement_started_step must be non-negative")
    ordered = list(points)
    for previous, current in zip(ordered, ordered[1:]):
        if current.step <= previous.step:
            raise ValueError("validation steps must be strictly increasing")
    step = current_step
    empty = ScheduleDecision(
        action="train",
        reason="collect_validation_history",
        step=step,
        recent_gain_per_update=None,
        reference_gain_per_update=None,
        gain_ratio=None,
        plateau_severity=0.0,
        replacement_fraction_cap=min_replacement_fraction,
    )
    if len(ordered) < window or current_step < min_step:
        if not probe_enabled:
            return empty

    if probe_enabled:
        assert refinement_started_step is not None
        assert pre_refinement_best is not None
        probe_end_step = refinement_started_step + probe_updates
        if current_step < probe_end_step:
            return ScheduleDecision(
                action="train",
                reason="probe_refinement_value",
                step=step,
                recent_gain_per_update=None,
                reference_gain_per_update=None,
                gain_ratio=None,
                plateau_severity=0.0,
                replacement_fraction_cap=min_replacement_fraction,
            )
        post_refinement_best = max(point.value for point in ordered)
        scale = max(abs(pre_refinement_best), 1e-12)
        relative_gain = (post_refinement_best - pre_refinement_best) / scale
        if relative_gain < min_relative_acceptance_gain:
            return ScheduleDecision(
                action="rollback",
                reason="refinement_probe_failed",
                step=step,
                recent_gain_per_update=None,
                reference_gain_per_update=None,
                gain_ratio=max(0.0, relative_gain),
                plateau_severity=1.0,
                replacement_fraction_cap=min_replacement_fraction,
            )

    if len(ordered) < window or current_step < min_step:
        return empty

    window_gains = [
        _window_gain(ordered[end - window : end])
        for end in range(window, len(ordered) + 1)
    ]
    recent_gain = window_gains[-1]
    reference_gain = max(window_gains)
    gain_ratio = (
        max(0.0, recent_gain) / reference_gain if reference_gain > 0 else 0.0
    )
    severity = min(1.0, max(0.0, 1.0 - gain_ratio))
    replacement_cap = min_replacement_fraction + severity * (
        max_replacement_fraction - min_replacement_fraction
    )
    common = dict(
        step=step,
        recent_gain_per_update=recent_gain,
        reference_gain_per_update=reference_gain,
        gain_ratio=gain_ratio,
        plateau_severity=severity,
        replacement_fraction_cap=replacement_cap,
    )
    if gain_ratio > trigger_gain_ratio:
        return ScheduleDecision(
            action="train", reason="validation_gain_is_still_efficient", **common
        )

    remaining = total_update_budget - current_step - refinements_used * scoring_cost_updates
    required = scoring_cost_updates + max(
        min_updates_after_refinement, probe_updates
    )
    if refinements_used < max_refinements and remaining >= required:
        return ScheduleDecision(
            action="refine", reason="validation_gain_plateaued", **common
        )
    reason = (
        "refinement_limit_reached"
        if refinements_used >= max_refinements
        else "insufficient_compute_runway"
    )
    return ScheduleDecision(action="stop", reason=reason, **common)


def adaptive_profitable_swap_fraction(
    swap_margins: Sequence[float],
    *,
    selectable_count: int,
    fraction_cap: float,
    min_standardized_margin: float = 0.05,
) -> float:
    """Return the fraction justified by positive, non-trivial swap margins."""
    if selectable_count <= 0:
        raise ValueError("selectable_count must be positive")
    if not 0 <= fraction_cap <= 1:
        raise ValueError("fraction_cap must be between zero and one")
    margins = [float(value) for value in swap_margins]
    if not margins or fraction_cap == 0:
        return 0.0
    mean = sum(margins) / len(margins)
    variance = sum((value - mean) ** 2 for value in margins) / len(margins)
    scale = max(variance**0.5, 1e-12)
    threshold = max(0.0, min_standardized_margin * scale)
    profitable = sum(value > threshold for value in margins)
    return min(fraction_cap, profitable / selectable_count)
