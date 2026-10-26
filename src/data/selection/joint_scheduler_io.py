from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from src.data.selection.joint_scheduler import CheckpointObservation


def load_per_user_checkpoint(
    path: str | Path, *, source: str
) -> tuple[torch.Tensor, CheckpointObservation]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("metric") != "ndcg@10":
        raise ValueError(f"{path} does not contain ndcg@10 observations")
    user_ids = torch.as_tensor(payload["user_ids"], dtype=torch.int64).flatten()
    values = torch.as_tensor(payload["values"], dtype=torch.float64).flatten()
    if not user_ids.numel() or user_ids.shape != values.shape:
        raise ValueError(f"{path} has invalid paired observations")
    if torch.unique(user_ids).numel() != user_ids.numel():
        raise ValueError(f"{path} contains duplicate user IDs")
    if not torch.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite NDCG values")
    order = torch.argsort(user_ids, stable=True)
    user_ids = user_ids[order]
    values = values[order]
    observation = CheckpointObservation.from_values(
        str(path), values.tolist(), source=source
    )
    return user_ids, observation


def require_aligned_users(records: Iterable[tuple[torch.Tensor, object]]) -> None:
    expected: torch.Tensor | None = None
    for user_ids, _ in records:
        if expected is None:
            expected = user_ids
        elif not torch.equal(expected, user_ids):
            raise ValueError("Action Trial observations do not contain identical users")
