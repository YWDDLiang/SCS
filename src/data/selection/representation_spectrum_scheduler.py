"""Streaming spectral scheduling for fixed-history augmentation views.

The scheduler only decides which auxiliary presentations receive the fixed
ordered-history transform.  It never chooses among augmentation operators.
Representations are supplied by the training data path and are projected into
a small, deterministic sketch before a covariance update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SpectrumScheduleResult:
    """Per-row spectral scores and the current diagnostic state."""

    scores: torch.Tensor
    projected: torch.Tensor
    effective_rank: torch.Tensor


class RepresentationSpectrumScheduler:
    """Online low-rank covariance scheduler with a fixed augmentation budget.

    ``score_and_observe`` scores a batch against the covariance accumulated
    before the batch and then folds the batch into that covariance.  The
    resulting scores can be used to allocate a fixed fraction of auxiliary
    slots to the most under-covered directions.
    """

    format_version = 1

    def __init__(
        self,
        projection_dim: int = 32,
        covariance_decay: float = 0.99,
        ridge: float = 1e-2,
        target_view_rate: float = 0.75,
        seed: int = 20261020,
    ) -> None:
        if projection_dim < 2:
            raise ValueError("projection_dim must be at least two")
        if not 0.0 < covariance_decay < 1.0:
            raise ValueError("covariance_decay must lie in (0, 1)")
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        if not 0.0 <= target_view_rate <= 1.0:
            raise ValueError("target_view_rate must lie in [0, 1]")
        self.projection_dim = int(projection_dim)
        self.covariance_decay = float(covariance_decay)
        self.ridge = float(ridge)
        self.target_view_rate = float(target_view_rate)
        self.seed = int(seed)
        self.projection: Optional[torch.Tensor] = None
        self.covariance: Optional[torch.Tensor] = None
        self.observed = 0
        self.refreshes = 0

    def _ensure(self, input_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.projection is not None:
            if self.projection.shape != (input_dim, self.projection_dim):
                raise ValueError("input dimension changed after scheduler initialization")
            if self.projection.device != device or self.projection.dtype != dtype:
                self.projection = self.projection.to(device=device, dtype=dtype)
            if self.covariance is not None and (
                self.covariance.device != device or self.covariance.dtype != dtype
            ):
                self.covariance = self.covariance.to(device=device, dtype=dtype)
            return
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        projection = torch.randn(
            input_dim, self.projection_dim, generator=generator, dtype=torch.float32
        )
        projection /= projection.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.projection = projection.to(device=device, dtype=dtype)
        self.covariance = torch.eye(
            self.projection_dim, device=device, dtype=dtype
        ) * self.ridge

    def _project(self, representations: torch.Tensor) -> torch.Tensor:
        if representations.ndim != 2 or representations.shape[0] == 0:
            raise ValueError("representations must be a non-empty [N, D] tensor")
        if not representations.is_floating_point() or not torch.isfinite(representations).all():
            raise ValueError("representations must be finite floating point values")
        self._ensure(representations.shape[1], representations.device, representations.dtype)
        assert self.projection is not None
        values = representations / representations.norm(dim=1, keepdim=True).clamp_min(1e-8)
        projected = values @ self.projection
        return projected / projected.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def _novelty(self, projected: torch.Tensor) -> torch.Tensor:
        assert self.covariance is not None
        covariance = (self.covariance + self.covariance.transpose(0, 1)) / 2
        regularized = covariance + self.ridge * torch.eye(
            covariance.shape[0], device=covariance.device, dtype=covariance.dtype
        )
        solved = torch.linalg.solve(regularized, projected.transpose(0, 1)).transpose(0, 1)
        return (projected * solved).sum(dim=1).clamp_min(0.0)

    def _rank(self) -> torch.Tensor:
        assert self.covariance is not None
        values = torch.linalg.eigvalsh(
            (self.covariance + self.covariance.transpose(0, 1)) / 2
        ).clamp_min(0.0)
        total = values.sum()
        if float(total) <= 1e-12:
            return torch.zeros((), device=values.device, dtype=values.dtype)
        probabilities = values / total
        return (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp()

    def score_and_observe(self, representations: torch.Tensor) -> SpectrumScheduleResult:
        """Score rows before updating the online spectrum, then observe them."""
        projected = self._project(representations)
        scores = self._novelty(projected)
        assert self.covariance is not None
        batch_covariance = projected.transpose(0, 1) @ projected / len(projected)
        self.covariance.mul_(self.covariance_decay).add_(
            batch_covariance, alpha=1.0 - self.covariance_decay
        )
        self.observed += int(len(projected))
        self.refreshes += 1
        return SpectrumScheduleResult(scores=scores, projected=projected, effective_rank=self._rank())

    def choose_views(
        self,
        scores: torch.Tensor,
        auxiliary: torch.Tensor,
        *,
        budget_size: Optional[int] = None,
        strata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Choose a fixed-rate subset of eligible auxiliary rows.

        ``budget_size`` lets callers express the rate against all auxiliary
        slots while masking out ineligible rows.  This preserves G's overall
        changed-slot budget when the raw history itself cannot be transformed.
        With ``strata``, quotas preserve the observed slot composition and
        systematic rank-spread selection covers the within-stratum spectrum
        instead of taking only the most novel rows.
        """
        if scores.ndim != 1 or auxiliary.ndim != 1 or len(scores) != len(auxiliary):
            raise ValueError("scores and auxiliary must be aligned one-dimensional tensors")
        if strata is not None and (strata.ndim != 1 or len(strata) != len(scores)):
            raise ValueError("strata must be aligned one-dimensional tensor")
        selected = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
        rows = torch.nonzero(auxiliary, as_tuple=False).flatten()
        if not len(rows) or self.target_view_rate <= 0.0:
            return selected
        denominator = len(rows) if budget_size is None else int(budget_size)
        if denominator < 0:
            raise ValueError("budget_size must be non-negative")
        count = int(round(self.target_view_rate * denominator))
        count = min(len(rows), max(0, count))
        if count and strata is None:
            ranked = torch.argsort(scores[rows], descending=True, stable=True)[:count]
            selected[rows[ranked]] = True
        elif count:
            # Preserve the observed eligible-slot composition, then use the
            # spectral residual only within each stratum.  This prevents a
            # novelty score from silently changing history-length exposure.
            groups, group_counts = torch.unique(
                strata[rows], sorted=True, return_counts=True
            )
            ideal = group_counts.to(dtype=scores.dtype) * (float(count) / len(rows))
            quotas = torch.floor(ideal).to(dtype=torch.long)
            remaining = count - int(quotas.sum())
            if remaining:
                remainder_order = torch.argsort(ideal - quotas.to(ideal.dtype), descending=True, stable=True)
                quotas[remainder_order[:remaining]] += 1
            for group, quota in zip(groups, quotas):
                quota = int(quota)
                if quota <= 0:
                    continue
                group_rows = rows[strata[rows] == group]
                ranked = torch.argsort(scores[group_rows], descending=True, stable=True)
                ranks = torch.floor(
                    (torch.arange(quota, device=scores.device, dtype=scores.dtype) + 0.5)
                    * (len(ranked) / quota)
                ).to(dtype=torch.long).clamp_max(len(ranked) - 1)
                selected[group_rows[ranked[ranks]]] = True
        return selected

    def state_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "projection_dim": self.projection_dim,
            "covariance_decay": self.covariance_decay,
            "ridge": self.ridge,
            "target_view_rate": self.target_view_rate,
            "seed": self.seed,
            "projection": None if self.projection is None else self.projection.detach().cpu(),
            "covariance": None if self.covariance is None else self.covariance.detach().cpu(),
            "observed": self.observed,
            "refreshes": self.refreshes,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("format_version") != self.format_version:
            raise RuntimeError("spectrum scheduler state format mismatch")
        for key in ("projection_dim", "seed"):
            if int(state[key]) != getattr(self, key):
                raise RuntimeError(f"spectrum scheduler {key} mismatch")
        self.projection = None if state["projection"] is None else state["projection"].clone()
        self.covariance = None if state["covariance"] is None else state["covariance"].clone()
        self.observed = int(state["observed"])
        self.refreshes = int(state["refreshes"])

