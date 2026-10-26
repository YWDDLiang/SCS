"""Online augmentation-view scheduling with LOOM and SPICE signals.

The scheduler operates on lightweight per-example gradient representations. It
does not generate views or compute full-model per-example gradients; callers
provide one representation for each ``(base_group, action)`` candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch


def _validate_vectors(
    representations: torch.Tensor,
    group_ids: torch.Tensor,
    action_ids: torch.Tensor,
) -> None:
    if representations.ndim != 2 or representations.shape[0] == 0:
        raise ValueError("representations must be a non-empty [N, D] tensor")
    if not representations.is_floating_point():
        raise ValueError("representations must be floating point")
    if group_ids.ndim != 1 or action_ids.ndim != 1:
        raise ValueError("group_ids and action_ids must be one-dimensional")
    if len(group_ids) != len(representations) or len(action_ids) != len(representations):
        raise ValueError("candidate vectors and ids must have equal length")
    if not torch.isfinite(representations).all():
        raise ValueError("representations must contain only finite values")
    if group_ids.device != representations.device or action_ids.device != representations.device:
        raise ValueError("representations and ids must be on the same device")


def _standardize(values: torch.Tensor) -> torch.Tensor:
    """Standardize a refresh-window signal without amplifying constants."""
    mean = values.mean()
    scale = values.std(unbiased=False)
    if not torch.isfinite(scale) or float(scale) <= 1e-12:
        scale = torch.ones((), device=values.device, dtype=values.dtype)
    return (values - mean) / scale


def group_leave_one_out_alignment(
    representations: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return LOOM alignment against candidates outside each base-example group.

    All views from the same base example are removed from the target. If a
    refresh window contains no outside group, the target is undefined and the
    corresponding alignment is zero rather than self-reinforcing the action.
    """
    if representations.ndim != 2 or group_ids.ndim != 1:
        raise ValueError("representations must be [N, D] and group_ids [N]")
    if len(representations) != len(group_ids) or len(representations) == 0:
        raise ValueError("representations and group_ids must be non-empty and aligned")
    if representations.device != group_ids.device:
        raise ValueError("representations and group_ids must be on the same device")
    if not torch.isfinite(representations).all():
        raise ValueError("representations must contain only finite values")
    _, inverse, counts = torch.unique(
        group_ids.to(torch.long), sorted=True, return_inverse=True, return_counts=True
    )
    group_sums = torch.zeros(
        (len(counts), representations.shape[1]),
        device=representations.device,
        dtype=representations.dtype,
    )
    group_sums.index_add_(0, inverse, representations)
    total = representations.sum(dim=0, keepdim=True)
    outside = total - group_sums[inverse]
    outside_count = len(representations) - counts[inverse]
    target = outside / outside_count.clamp_min(1).to(representations.dtype).unsqueeze(1)
    numerator = (representations * target).sum(dim=1)
    denominator = representations.norm(dim=1) * target.norm(dim=1)
    alignment = numerator / denominator.clamp_min(eps)
    return torch.where(outside_count > 0, alignment, torch.zeros_like(alignment))


def spectral_novelty(
    representations: torch.Tensor,
    covariance: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute SPICE's rank-one Fisher/log-det marginal gain."""
    if representations.ndim != 2 or covariance.ndim != 2:
        raise ValueError("representations must be [N, D] and covariance [D, D]")
    if covariance.shape != (representations.shape[1], representations.shape[1]):
        raise ValueError("covariance dimension does not match representations")
    if covariance.device != representations.device or covariance.dtype != representations.dtype:
        raise ValueError("representations and covariance must have the same device and dtype")
    if not torch.isfinite(covariance).all():
        raise ValueError("covariance must contain only finite values")
    covariance = (covariance + covariance.transpose(0, 1)) / 2
    # F is regularized and updated by positive semidefinite outer products.
    # A tiny diagonal fallback keeps the probe robust to low-precision rounding.
    try:
        chol = torch.linalg.cholesky(covariance)
    except RuntimeError:
        chol = torch.linalg.cholesky(covariance + eps * torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype))
    solved = torch.cholesky_solve(representations.unsqueeze(-1), chol).squeeze(-1)
    quadratic = (representations * solved).sum(dim=1).clamp_min(0)
    return torch.log1p(quadratic)


def gradient_conflict(
    representations: torch.Tensor,
    direction: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return SPICE's positive negative-cosine conflict penalty."""
    if representations.ndim != 2 or direction.ndim != 1:
        raise ValueError("representations must be [N, D] and direction [D]")
    if direction.shape[0] != representations.shape[1]:
        raise ValueError("direction dimension does not match representations")
    if direction.device != representations.device or direction.dtype != representations.dtype:
        raise ValueError("representations and direction must have the same device and dtype")
    cosine = (representations * direction.unsqueeze(0)).sum(dim=1)
    cosine = cosine / (representations.norm(dim=1) * direction.norm()).clamp_min(eps)
    return (-cosine).clamp_min(0)


def effective_rank(covariance: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Return the entropy effective rank used as a spectrum diagnostic."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not torch.isfinite(covariance).all():
        raise ValueError("covariance must contain only finite values")
    symmetric = (covariance + covariance.transpose(0, 1)) / 2
    eigenvalues = torch.linalg.eigvalsh(symmetric).clamp_min(0)
    total = eigenvalues.sum()
    if float(total) <= eps:
        return torch.zeros((), device=covariance.device, dtype=covariance.dtype)
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    return entropy.exp()


@dataclass(frozen=True)
class ViewRefreshResult:
    """Signals and action probabilities produced by one scheduler refresh."""

    group_ids: torch.Tensor
    action_ids: torch.Tensor
    loom_alignment: torch.Tensor
    novelty: torch.Tensor
    conflict: torch.Tensor
    combined_score: torch.Tensor
    probabilities: torch.Tensor
    policy_groups: torch.Tensor


class LOOMSPICEViewScheduler:
    """Maintain an online soft policy over label-preserving view actions.

    ``refresh`` expects exactly one candidate for every action in every group.
    This makes the policy auditable and prevents a missing action from silently
    changing the normalization. ``observe`` receives representations from the
    actions actually consumed by training and updates the Fisher state.
    """

    def __init__(
        self,
        num_actions: int,
        *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        conflict_weight: float = 1.0,
        cost_weight: float = 0.0,
        temperature: float = 1.0,
        exploration: float = 0.05,
        min_identity_probability: float = 0.05,
        identity_action: int = 0,
        score_ema_decay: float = 0.9,
        ridge: float = 1e-3,
        prior: Optional[Sequence[float]] = None,
    ) -> None:
        if num_actions < 2:
            raise ValueError("num_actions must be at least two")
        if not 0 < temperature:
            raise ValueError("temperature must be positive")
        if not 0 <= exploration <= 1:
            raise ValueError("exploration must be in [0, 1]")
        if not 0 <= min_identity_probability < 1:
            raise ValueError("min_identity_probability must be in [0, 1)")
        if not 0 <= score_ema_decay < 1:
            raise ValueError("score_ema_decay must be in [0, 1)")
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        if identity_action < 0 or identity_action >= num_actions:
            raise ValueError("identity_action must index an action")
        self.num_actions = int(num_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.conflict_weight = float(conflict_weight)
        self.cost_weight = float(cost_weight)
        self.temperature = float(temperature)
        self.exploration = float(exploration)
        self.min_identity_probability = float(min_identity_probability)
        self.identity_action = int(identity_action)
        self.score_ema_decay = float(score_ema_decay)
        self.ridge = float(ridge)
        if prior is None:
            prior = [1.0] * self.num_actions
        if len(prior) != self.num_actions or any(float(value) <= 0 for value in prior):
            raise ValueError("prior must contain one positive value per action")
        prior_tensor = torch.as_tensor(prior, dtype=torch.float32)
        self.prior = prior_tensor / prior_tensor.sum()
        self.covariance: Optional[torch.Tensor] = None
        self.direction: Optional[torch.Tensor] = None
        self._ema_scores: dict[int, torch.Tensor] = {}
        self._policy: dict[int, torch.Tensor] = {}
        self.last_refresh_step: Optional[int] = None

    @property
    def feature_dim(self) -> Optional[int]:
        return None if self.covariance is None else int(self.covariance.shape[0])

    def _ensure_state(self, feature_dim: int, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.covariance is None:
            self.covariance = self.ridge * torch.eye(feature_dim, device=device, dtype=dtype)
            self.direction = torch.zeros(feature_dim, device=device, dtype=dtype)
            self.prior = self.prior.to(device=device, dtype=dtype)
            return
        if self.feature_dim != feature_dim:
            raise ValueError("feature dimension changed after scheduler initialization")
        if self.covariance.device != device or self.covariance.dtype != dtype:
            self.covariance = self.covariance.to(device=device, dtype=dtype)
            self.direction = self.direction.to(device=device, dtype=dtype)
            self.prior = self.prior.to(device=device, dtype=dtype)

    def observe(self, representations: torch.Tensor, *, decay: Optional[float] = None) -> None:
        """Update Fisher covariance and EMA direction from consumed actions."""
        if representations.ndim != 2 or representations.shape[0] == 0:
            raise ValueError("observed representations must be non-empty [N, D]")
        if not representations.is_floating_point() or not torch.isfinite(representations).all():
            raise ValueError("observed representations must be finite floating point")
        self._ensure_state(representations.shape[1], device=representations.device, dtype=representations.dtype)
        assert self.covariance is not None and self.direction is not None
        self.covariance.add_(representations.transpose(0, 1) @ representations)
        observed_direction = representations.mean(dim=0)
        rate = self.score_ema_decay if decay is None else float(decay)
        if not 0 <= rate < 1:
            raise ValueError("decay must be in [0, 1)")
        self.direction.mul_(rate).add_(observed_direction, alpha=1 - rate)

    def _validate_action_grid(self, group_ids: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        groups = torch.unique(group_ids.to(torch.long), sorted=True)
        for group in groups.tolist():
            actions = action_ids[group_ids == group].to(torch.long)
            if len(actions) != self.num_actions or torch.unique(actions).numel() != self.num_actions:
                raise ValueError("each group must contain each action exactly once")
            if actions.min() < 0 or actions.max() >= self.num_actions:
                raise ValueError("action ids must index the configured action set")
        return groups

    def _apply_identity_floor(self, probabilities: torch.Tensor) -> torch.Tensor:
        floor = self.min_identity_probability
        if floor <= 0:
            return probabilities
        current = probabilities[:, self.identity_action]
        needs_floor = current < floor
        if not bool(needs_floor.any()):
            return probabilities
        result = probabilities.clone()
        for row in torch.nonzero(needs_floor, as_tuple=False).flatten().tolist():
            old = float(result[row, self.identity_action])
            result[row, self.identity_action] = floor
            remainder = max(1.0 - old, 1e-12)
            result[row] *= (1.0 - floor) / remainder
            result[row, self.identity_action] = floor
        return result

    def refresh(
        self,
        representations: torch.Tensor,
        group_ids: torch.Tensor,
        action_ids: torch.Tensor,
        *,
        costs: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> ViewRefreshResult:
        """Score a refresh window and update the per-group action policy."""
        _validate_vectors(representations, group_ids, action_ids)
        group_ids = group_ids.to(torch.long)
        action_ids = action_ids.to(torch.long)
        groups = self._validate_action_grid(group_ids, action_ids)
        self._ensure_state(representations.shape[1], device=representations.device, dtype=representations.dtype)
        assert self.covariance is not None and self.direction is not None
        if costs is None:
            costs = torch.zeros(len(representations), device=representations.device, dtype=representations.dtype)
        costs = costs.to(device=representations.device, dtype=representations.dtype)
        if costs.ndim == 0:
            costs = costs.expand(len(representations))
        if costs.shape != (len(representations),) or not torch.isfinite(costs).all() or bool((costs < 0).any()):
            raise ValueError("costs must be a finite non-negative vector aligned to candidates")
        loom = group_leave_one_out_alignment(representations, group_ids)
        novelty = spectral_novelty(representations, self.covariance)
        conflict = gradient_conflict(representations, self.direction)
        combined = (
            self.alpha * _standardize(novelty)
            + self.gamma * _standardize(loom)
            - self.conflict_weight * _standardize(conflict)
            - self.cost_weight * _standardize(costs)
        )
        score_rows = torch.stack([
            combined[group_ids == group].index_select(0, torch.argsort(action_ids[group_ids == group]))
            for group in groups
        ])
        group_keys = [int(group) for group in groups.tolist()]
        smoothed_rows = []
        for key, row in zip(group_keys, score_rows):
            previous = self._ema_scores.get(key)
            if previous is None:
                smoothed = row.detach().clone()
            else:
                previous = previous.to(device=row.device, dtype=row.dtype)
                smoothed = self.score_ema_decay * previous + (1 - self.score_ema_decay) * row
            self._ema_scores[key] = smoothed.detach().clone()
            smoothed_rows.append(smoothed)
        smoothed_scores = torch.stack(smoothed_rows)
        logits = (smoothed_scores + self.prior.log().unsqueeze(0)) / self.temperature
        probabilities = torch.softmax(logits, dim=1)
        probabilities = (1 - self.exploration) * probabilities + self.exploration / self.num_actions
        probabilities = self._apply_identity_floor(probabilities)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        for key, row in zip(group_keys, probabilities):
            self._policy[key] = row.detach().clone()
        self.last_refresh_step = None if step is None else int(step)
        candidate_probabilities = torch.empty(len(representations), device=representations.device, dtype=representations.dtype)
        for row, group in enumerate(groups.tolist()):
            mask = group_ids == group
            candidate_probabilities[mask] = probabilities[row, action_ids[mask]]
        return ViewRefreshResult(
            group_ids=group_ids,
            action_ids=action_ids,
            loom_alignment=loom,
            novelty=novelty,
            conflict=conflict,
            combined_score=combined,
            probabilities=candidate_probabilities,
            policy_groups=groups,
        )

    def sample_actions(
        self,
        group_ids: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample one action per base group using the latest refresh policy."""
        if group_ids.ndim != 1:
            raise ValueError("group_ids must be one-dimensional")
        probabilities = self.action_probabilities(group_ids)
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(1)

    def action_probabilities(self, group_ids: torch.Tensor) -> torch.Tensor:
        """Return the latest ``[batch, num_actions]`` policy rows for groups."""
        if group_ids.ndim != 1:
            raise ValueError("group_ids must be one-dimensional")
        fallback = self.prior.to(device=group_ids.device)
        return torch.stack([
            self._policy[int(group)].to(device=group_ids.device)
            if int(group) in self._policy
            else fallback
            for group in group_ids.detach().cpu().tolist()
        ])

    def state_dict(self) -> dict[str, object]:
        """Return a checkpoint-friendly scheduler state."""
        return {
            "format_version": 1,
            "num_actions": self.num_actions,
            "config": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "conflict_weight": self.conflict_weight,
                "cost_weight": self.cost_weight,
                "temperature": self.temperature,
                "exploration": self.exploration,
                "min_identity_probability": self.min_identity_probability,
                "identity_action": self.identity_action,
                "score_ema_decay": self.score_ema_decay,
                "ridge": self.ridge,
            },
            "covariance": None if self.covariance is None else self.covariance.detach().cpu(),
            "direction": None if self.direction is None else self.direction.detach().cpu(),
            "ema_scores": {key: value.detach().cpu() for key, value in self._ema_scores.items()},
            "policy": {key: value.detach().cpu() for key, value in self._policy.items()},
            "last_refresh_step": self.last_refresh_step,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state.get("format_version", -1)) != 1:
            raise ValueError("unsupported LOOM-SPICE scheduler state format")
        if int(state.get("num_actions", -1)) != self.num_actions:
            raise ValueError("scheduler state action count does not match configuration")
        saved_config = state.get("config")
        if saved_config is not None:
            if not isinstance(saved_config, Mapping):
                raise ValueError("scheduler state config must be a mapping")
            current_config = self.state_dict()["config"]
            assert isinstance(current_config, dict)
            if any(float(saved_config[key]) != float(current_config[key]) for key in current_config if key != "identity_action"):
                raise ValueError("scheduler state hyperparameters do not match configuration")
            if int(saved_config["identity_action"]) != self.identity_action:
                raise ValueError("scheduler state identity action does not match configuration")
        covariance = state.get("covariance")
        direction = state.get("direction")
        self.covariance = None if covariance is None else torch.as_tensor(covariance).clone()
        self.direction = None if direction is None else torch.as_tensor(direction).clone()
        if (self.covariance is None) != (self.direction is None):
            raise ValueError("scheduler covariance and direction must be present together")
        if self.covariance is not None and self.covariance.ndim != 2:
            raise ValueError("scheduler covariance must be a matrix")
        if self.direction is not None and (self.direction.ndim != 1 or self.direction.shape[0] != self.covariance.shape[0]):
            raise ValueError("scheduler direction does not match covariance")
        self._ema_scores = {int(key): torch.as_tensor(value).clone() for key, value in dict(state.get("ema_scores", {})).items()}
        self._policy = {int(key): torch.as_tensor(value).clone() for key, value in dict(state.get("policy", {})).items()}
        self.last_refresh_step = state.get("last_refresh_step")
