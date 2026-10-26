"""Allocate history views using spectrum priorities and gradient conflict.

The policy receives one spectrum score and one conflict score per event.
Spectrum priorities determine which events receive additional presentations;
the output-head conflict proxy guides the repeat-versus-view allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class PolicySample:
    """Sampled auxiliary events and their action probabilities."""

    event_indices: torch.Tensor
    augment: torch.Tensor
    event_probabilities: torch.Tensor
    augmentation_probabilities: torch.Tensor
    target_augmentation_rate: torch.Tensor


class ConflictAwarePresentationPolicy:
    """Allocate history views without changing prediction targets.

    A refresh-window conflict rate determines a bounded target augmentation
    mass. Within the spectrum-weighted event distribution, larger positive
    conflict receives a larger Bernoulli probability of the ordered-history
    action; remaining mass is literal repetition. Sampling is with replacement
    so a fixed event pool can represent the additional presentation budget.
    """

    format_version = 1

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        min_augmentation_rate: float = 0.50,
        max_augmentation_rate: float = 0.90,
        seed: int = 20261020,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= min_augmentation_rate <= max_augmentation_rate <= 1:
            raise ValueError("augmentation-rate bounds must satisfy 0 <= min <= max <= 1")
        self.temperature = float(temperature)
        self.min_augmentation_rate = float(min_augmentation_rate)
        self.max_augmentation_rate = float(max_augmentation_rate)
        self.seed = int(seed)

    @staticmethod
    def _finite_vector(values: torch.Tensor, name: str) -> torch.Tensor:
        values = torch.as_tensor(values, dtype=torch.float32).flatten()
        if values.numel() == 0 or not torch.isfinite(values).all():
            raise ValueError(f"{name} must be a non-empty finite vector")
        return values

    def event_probabilities(self, spectrum_scores: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled event probabilities from spectrum scores."""
        scores = self._finite_vector(spectrum_scores, "spectrum_scores")
        centered = scores - scores.mean()
        logits = (centered / (scores.std(unbiased=False).clamp_min(1e-6))) / self.temperature
        return torch.softmax(logits, dim=0)

    def augmentation_probabilities(
        self,
        conflict_scores: torch.Tensor,
        event_probabilities: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map positive conflict to action probabilities with exact expected mass.

        The target rate is the unweighted positive-conflict rate, bounded by
        the configured limits. Probabilities are calibrated under the supplied
        spectrum event distribution, so the expected augmentation fraction of
        sampled auxiliary presentations equals that target (up to float error).
        """
        conflict = self._finite_vector(conflict_scores, "conflict_scores").clamp_min(0.0)
        conflict_rate = (conflict > 0).to(torch.float32).mean()
        target = self.min_augmentation_rate + conflict_rate * (
            self.max_augmentation_rate - self.min_augmentation_rate
        )
        if event_probabilities is None:
            weights = torch.full_like(conflict, 1.0 / len(conflict))
        else:
            weights = self._finite_vector(event_probabilities, "event_probabilities")
            if weights.shape != conflict.shape or (weights < 0).any():
                raise ValueError("event_probabilities must be non-negative and aligned with conflict_scores")
            weight_sum = weights.sum()
            if float(weight_sum) <= 1e-12:
                raise ValueError("event_probabilities must have positive total mass")
            weights = weights / weight_sum
        if float(conflict.max()) <= 1e-12 or float(conflict.max() - conflict.min()) <= 1e-12:
            return torch.full_like(conflict, target), target

        normalized = conflict / conflict.max()
        # An additive offset allows the requested rate to be reached even
        # when many events have zero conflict, while keeping probabilities
        # monotone in conflict. Binary search calibrates the spectrum-weighted
        # expected action mass exactly.
        low = -1.0
        high = 1.0
        for _ in range(48):
            offset = (low + high) / 2.0
            probabilities = (normalized + offset).clamp(0.0, 1.0)
            expected = (probabilities * weights).sum()
            if float(expected) < float(target):
                low = offset
            else:
                high = offset
        probabilities = (normalized + ((low + high) / 2.0)).clamp(0.0, 1.0)
        return probabilities, target

    def sample(
        self,
        spectrum_scores: torch.Tensor,
        conflict_scores: torch.Tensor,
        num_slots: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> PolicySample:
        """Sample ``num_slots`` auxiliary presentations with replacement."""
        spectrum = self._finite_vector(spectrum_scores, "spectrum_scores")
        conflict = self._finite_vector(conflict_scores, "conflict_scores")
        if spectrum.shape != conflict.shape:
            raise ValueError("spectrum_scores and conflict_scores must have equal shape")
        if num_slots < 0:
            raise ValueError("num_slots must be non-negative")
        probabilities = self.event_probabilities(spectrum)
        action_probabilities, target = self.augmentation_probabilities(conflict, probabilities)
        if num_slots == 0:
            empty = torch.empty(0, dtype=torch.long, device=probabilities.device)
            return PolicySample(empty, empty.to(torch.bool), probabilities, action_probabilities, target)
        event_indices = torch.multinomial(
            probabilities,
            num_slots,
            replacement=True,
            generator=generator,
        )
        augment = torch.rand(
            num_slots,
            device=probabilities.device,
            generator=generator,
        ) < action_probabilities[event_indices]
        return PolicySample(
            event_indices=event_indices,
            augment=augment,
            event_probabilities=probabilities,
            augmentation_probabilities=action_probabilities,
            target_augmentation_rate=target,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "temperature": self.temperature,
            "min_augmentation_rate": self.min_augmentation_rate,
            "max_augmentation_rate": self.max_augmentation_rate,
            "seed": self.seed,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state.get("format_version", -1)) != self.format_version:
            raise ValueError("unsupported ConflictAwarePresentationPolicy state format")
        for key in ("temperature", "min_augmentation_rate", "max_augmentation_rate", "seed"):
            if float(state[key]) != float(getattr(self, key)):
                raise ValueError(f"policy configuration mismatch for {key}")
