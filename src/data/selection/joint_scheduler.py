from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Sequence

import torch


Action = Literal["continue", "reselect"]
NextStep = Literal["action_trial", "continue_remainder", "stop"]


def _require_nonnegative_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ComputeBreakdown:
    """Mutually exclusive compute charges plus resource measurements."""

    training_ue: float = 0.0
    selection_ue: float = 0.0
    evaluation_ue: float = 0.0
    control_ue: float = 0.0
    gpu_device_seconds: float = 0.0
    cpu_core_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("training_ue", self.training_ue),
            ("selection_ue", self.selection_ue),
            ("evaluation_ue", self.evaluation_ue),
            ("control_ue", self.control_ue),
            ("gpu_device_seconds", self.gpu_device_seconds),
            ("cpu_core_seconds", self.cpu_core_seconds),
        ):
            _require_nonnegative_finite(float(value), name)

    @property
    def total_ue(self) -> float:
        return (
            self.training_ue
            + self.selection_ue
            + self.evaluation_ue
            + self.control_ue
        )

    def as_control(self) -> ComputeBreakdown:
        """Reclassify a discarded arm without double-counting its work."""
        return ComputeBreakdown(
            control_ue=self.total_ue,
            gpu_device_seconds=self.gpu_device_seconds,
            cpu_core_seconds=self.cpu_core_seconds,
        )

    def __add__(self, other: ComputeBreakdown) -> ComputeBreakdown:
        if not isinstance(other, ComputeBreakdown):
            return NotImplemented
        return ComputeBreakdown(
            training_ue=self.training_ue + other.training_ue,
            selection_ue=self.selection_ue + other.selection_ue,
            evaluation_ue=self.evaluation_ue + other.evaluation_ue,
            control_ue=self.control_ue + other.control_ue,
            gpu_device_seconds=self.gpu_device_seconds
            + other.gpu_device_seconds,
            cpu_core_seconds=self.cpu_core_seconds + other.cpu_core_seconds,
        )


@dataclass(frozen=True)
class CheckpointObservation:
    checkpoint_id: str
    per_user_ndcg10: tuple[float, ...]
    source: str

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must not be empty")
        if not self.per_user_ndcg10:
            raise ValueError("per_user_ndcg10 must not be empty")
        if not all(math.isfinite(value) for value in self.per_user_ndcg10):
            raise ValueError("per_user_ndcg10 must contain only finite values")

    @classmethod
    def from_values(
        cls,
        checkpoint_id: str,
        per_user_ndcg10: Iterable[float],
        *,
        source: str,
    ) -> CheckpointObservation:
        return cls(
            checkpoint_id=checkpoint_id,
            per_user_ndcg10=tuple(float(value) for value in per_user_ndcg10),
            source=source,
        )

    @property
    def value(self) -> float:
        return sum(self.per_user_ndcg10) / len(self.per_user_ndcg10)


class CheckpointBank:
    def __init__(self) -> None:
        self._observations: dict[str, CheckpointObservation] = {}

    def add(self, observations: Iterable[CheckpointObservation]) -> None:
        for observation in observations:
            existing = self._observations.get(observation.checkpoint_id)
            if existing is not None and existing != observation:
                raise ValueError(
                    f"conflicting checkpoint observation: {observation.checkpoint_id}"
                )
            self._observations[observation.checkpoint_id] = observation

    def best(self) -> CheckpointObservation:
        if not self._observations:
            raise RuntimeError("Checkpoint Bank is empty")
        return min(
            self._observations.values(),
            key=lambda observation: (-observation.value, observation.checkpoint_id),
        )

    def __len__(self) -> int:
        return len(self._observations)


@dataclass(frozen=True)
class TrialArm:
    action: Action
    endpoint_id: str
    checkpoints: tuple[CheckpointObservation, ...]
    costs: ComputeBreakdown
    allowance_ue: float
    feasible: bool = True

    def __post_init__(self) -> None:
        _require_nonnegative_finite(self.allowance_ue, "allowance_ue")
        if not self.endpoint_id:
            raise ValueError("endpoint_id must not be empty")
        if self.costs.total_ue > self.allowance_ue + 1e-9:
            raise ValueError("arm cost exceeds its Trial Allowance")
        if self.feasible and not self.checkpoints:
            raise ValueError("a feasible arm must have an evaluated checkpoint")

    def best(self) -> CheckpointObservation:
        if not self.feasible or not self.checkpoints:
            raise RuntimeError("an infeasible arm has no best checkpoint")
        return min(
            self.checkpoints,
            key=lambda observation: (-observation.value, observation.checkpoint_id),
        )


@dataclass(frozen=True)
class TrialDecision:
    action: Action
    winning_endpoint_id: str
    reselect_minus_continue_lower_bound: float | None
    continue_improves_bank: bool
    reselect_improves_bank: bool
    stop_evidence: bool
    reason: str


@dataclass(frozen=True)
class SchedulerProgress:
    consecutive_failures: int = 0

    def __post_init__(self) -> None:
        if self.consecutive_failures < 0:
            raise ValueError("consecutive_failures must be non-negative")

    def after_trial(self, *, stop_evidence: bool) -> SchedulerProgress:
        return SchedulerProgress(
            self.consecutive_failures + 1 if stop_evidence else 0
        )

    @property
    def should_stop(self) -> bool:
        return self.consecutive_failures >= 2


@dataclass(frozen=True)
class ReselectBudget:
    training_ue: float
    feasible: bool


def plan_next_step(
    remaining_ue: float,
    trial_allowance_ue: float,
    *,
    final_evaluation_reserve_ue: float = 0.0,
) -> NextStep:
    for value, name in (
        (remaining_ue, "remaining_ue"),
        (trial_allowance_ue, "trial_allowance_ue"),
        (final_evaluation_reserve_ue, "final_evaluation_reserve_ue"),
    ):
        _require_nonnegative_finite(value, name)
    if trial_allowance_ue <= 0:
        raise ValueError("trial_allowance_ue must be positive")
    if remaining_ue + 1e-9 >= 2 * trial_allowance_ue:
        return "action_trial"
    if remaining_ue + 1e-9 >= final_evaluation_reserve_ue:
        return "continue_remainder"
    return "stop"


def allocate_reselect_training_budget(
    trial_allowance_ue: float,
    selection_ue: float,
    evaluation_reserve_ue: float,
) -> ReselectBudget:
    for value, name in (
        (trial_allowance_ue, "trial_allowance_ue"),
        (selection_ue, "selection_ue"),
        (evaluation_reserve_ue, "evaluation_reserve_ue"),
    ):
        _require_nonnegative_finite(value, name)
    if trial_allowance_ue <= 0:
        raise ValueError("trial_allowance_ue must be positive")
    training_ue = trial_allowance_ue - selection_ue - evaluation_reserve_ue
    return ReselectBudget(training_ue=max(0.0, training_ue), feasible=training_ue > 0)


def paired_bootstrap_lower_bound(
    challenger: Sequence[float],
    baseline: Sequence[float],
    *,
    confidence: float = 0.90,
    samples: int = 2000,
    seed: int = 0,
    batch_size: int = 128,
) -> float:
    """Return a one-sided lower confidence bound for a paired mean gain."""
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be within (0.5, 1)")
    if samples <= 0 or batch_size <= 0:
        raise ValueError("samples and batch_size must be positive")
    challenger_tensor = torch.as_tensor(challenger, dtype=torch.float64).flatten()
    baseline_tensor = torch.as_tensor(baseline, dtype=torch.float64).flatten()
    if challenger_tensor.shape != baseline_tensor.shape or not challenger_tensor.numel():
        raise ValueError("paired inputs must have equal, non-zero length")
    if not torch.isfinite(challenger_tensor).all() or not torch.isfinite(
        baseline_tensor
    ).all():
        raise ValueError("paired inputs must contain only finite values")

    differences = challenger_tensor - baseline_tensor
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    means: list[torch.Tensor] = []
    remaining = samples
    while remaining:
        current_batch = min(batch_size, remaining)
        indices = torch.randint(
            differences.numel(),
            (current_batch, differences.numel()),
            generator=generator,
        )
        means.append(differences[indices].mean(dim=1))
        remaining -= current_batch
    bootstrap_means = torch.cat(means)
    return float(torch.quantile(bootstrap_means, 1.0 - confidence))


def _significantly_improves(
    challenger: CheckpointObservation,
    baseline: CheckpointObservation,
    *,
    confidence: float,
    samples: int,
    seed: int,
) -> bool:
    return (
        paired_bootstrap_lower_bound(
            challenger.per_user_ndcg10,
            baseline.per_user_ndcg10,
            confidence=confidence,
            samples=samples,
            seed=seed,
        )
        > 0
    )


def decide_action_trial(
    continue_arm: TrialArm,
    reselect_arm: TrialArm,
    *,
    bank_best_before_trial: CheckpointObservation | None,
    confidence: float = 0.90,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> TrialDecision:
    if continue_arm.action != "continue" or reselect_arm.action != "reselect":
        raise ValueError("trial arms must be Continue then Reselect")
    if abs(continue_arm.allowance_ue - reselect_arm.allowance_ue) > 1e-9:
        raise ValueError("Action Trial arms must have equal allowances")

    best_continue = continue_arm.best()
    best_reselect = reselect_arm.best() if reselect_arm.feasible else None
    lower_bound: float | None = None
    if best_reselect is not None:
        lower_bound = paired_bootstrap_lower_bound(
            best_reselect.per_user_ndcg10,
            best_continue.per_user_ndcg10,
            confidence=confidence,
            samples=bootstrap_samples,
            seed=seed,
        )

    if lower_bound is not None and lower_bound > 0:
        action: Action = "reselect"
        endpoint_id = reselect_arm.endpoint_id
        reason = "reselect_reliably_beats_continue"
    else:
        action = "continue"
        endpoint_id = continue_arm.endpoint_id
        reason = (
            "reselect_infeasible"
            if best_reselect is None
            else "reselect_gain_is_uncertain"
        )

    if bank_best_before_trial is None:
        continue_improves = True
        reselect_improves = best_reselect is not None
        stop_evidence = False
    else:
        continue_improves = _significantly_improves(
            best_continue,
            bank_best_before_trial,
            confidence=confidence,
            samples=bootstrap_samples,
            seed=seed + 1,
        )
        reselect_improves = best_reselect is not None and _significantly_improves(
            best_reselect,
            bank_best_before_trial,
            confidence=confidence,
            samples=bootstrap_samples,
            seed=seed + 2,
        )
        stop_evidence = not continue_improves and not reselect_improves

    return TrialDecision(
        action=action,
        winning_endpoint_id=endpoint_id,
        reselect_minus_continue_lower_bound=lower_bound,
        continue_improves_bank=continue_improves,
        reselect_improves_bank=reselect_improves,
        stop_evidence=stop_evidence,
        reason=reason,
    )


def settle_trial_costs(
    continue_costs: ComputeBreakdown,
    reselect_costs: ComputeBreakdown,
    *,
    winner: Action,
    comparison_control_ue: float = 0.0,
    comparison_gpu_device_seconds: float = 0.0,
    comparison_cpu_core_seconds: float = 0.0,
) -> ComputeBreakdown:
    for value, name in (
        (comparison_control_ue, "comparison_control_ue"),
        (comparison_gpu_device_seconds, "comparison_gpu_device_seconds"),
        (comparison_cpu_core_seconds, "comparison_cpu_core_seconds"),
    ):
        _require_nonnegative_finite(value, name)
    comparison = ComputeBreakdown(
        control_ue=comparison_control_ue,
        gpu_device_seconds=comparison_gpu_device_seconds,
        cpu_core_seconds=comparison_cpu_core_seconds,
    )
    if winner == "continue":
        return continue_costs + reselect_costs.as_control() + comparison
    if winner == "reselect":
        return reselect_costs + continue_costs.as_control() + comparison
    raise ValueError(f"unknown winner: {winner}")
