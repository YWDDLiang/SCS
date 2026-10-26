from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset


FORMAT_VERSION = 1


def _as_flat_int_tensor(value: torch.Tensor | Sequence[int]) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.int32).flatten().contiguous()


@dataclass
class TargetPositionStore:
    """Compact user-sequence storage plus selectable target-position pairs."""

    tokens: torch.Tensor
    offsets: torch.Tensor
    user_ids: torch.Tensor
    target_pairs: torch.Tensor
    sid_hierarchy: int
    sequence_length: int
    metadata: Mapping[str, Any]

    @property
    def num_rows(self) -> int:
        return int(self.offsets.numel() - 1)

    @property
    def num_targets(self) -> int:
        return int(self.target_pairs.shape[0])

    @property
    def max_items(self) -> int:
        return self.sequence_length // self.sid_hierarchy

    @classmethod
    def from_sequences(
        cls,
        sequences: Iterable[torch.Tensor | Sequence[int]],
        user_ids: Iterable[int],
        *,
        sid_hierarchy: int,
        sequence_length: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "TargetPositionStore":
        if sid_hierarchy <= 0:
            raise ValueError("sid_hierarchy must be positive")
        if sequence_length < 2 * sid_hierarchy:
            raise ValueError("sequence_length must fit a history item and a target")
        if sequence_length % sid_hierarchy:
            raise ValueError("sequence_length must be divisible by sid_hierarchy")

        sequence_list = [_as_flat_int_tensor(sequence) for sequence in sequences]
        user_id_list = [int(user_id) for user_id in user_ids]
        if len(sequence_list) != len(user_id_list):
            raise ValueError("sequences and user_ids must have the same length")

        offsets = [0]
        target_pairs = []
        for user_row, sequence in enumerate(sequence_list):
            if sequence.numel() % sid_hierarchy:
                raise ValueError(
                    f"row {user_row} has {sequence.numel()} tokens, which is not "
                    f"divisible by sid_hierarchy={sid_hierarchy}"
                )
            item_count = sequence.numel() // sid_hierarchy
            offsets.append(offsets[-1] + sequence.numel())
            target_pairs.extend((user_row, target_position) for target_position in range(1, item_count))

        tokens = (
            torch.cat(sequence_list)
            if sequence_list
            else torch.empty(0, dtype=torch.int32)
        )
        pairs = torch.tensor(target_pairs, dtype=torch.int32).reshape(-1, 2)
        store = cls(
            tokens=tokens,
            offsets=torch.tensor(offsets, dtype=torch.int64),
            user_ids=torch.tensor(user_id_list, dtype=torch.int64),
            target_pairs=pairs,
            sid_hierarchy=sid_hierarchy,
            sequence_length=sequence_length,
            metadata=dict(metadata or {}),
        )
        store.validate()
        return store

    def validate(self) -> None:
        if self.sid_hierarchy <= 0:
            raise ValueError("sid_hierarchy must be positive")
        if self.sequence_length % self.sid_hierarchy:
            raise ValueError("sequence_length must be divisible by sid_hierarchy")
        if self.offsets.ndim != 1 or self.offsets.numel() == 0:
            raise ValueError("offsets must be a non-empty one-dimensional tensor")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != self.tokens.numel():
            raise ValueError("offsets must span the complete token tensor")
        if torch.any(self.offsets[1:] < self.offsets[:-1]):
            raise ValueError("offsets must be monotonic")
        if self.user_ids.numel() != self.num_rows:
            raise ValueError("user_ids must contain one value per user row")
        if self.target_pairs.ndim != 2 or self.target_pairs.shape[1] != 2:
            raise ValueError("target_pairs must have shape [N, 2]")

        for user_row, target_position in self.target_pairs.tolist():
            if not 0 <= user_row < self.num_rows:
                raise ValueError(f"invalid user_row {user_row}")
            token_count = int(self.offsets[user_row + 1] - self.offsets[user_row])
            item_count = token_count // self.sid_hierarchy
            if not 1 <= target_position < item_count:
                raise ValueError(
                    f"invalid target_position {target_position} for row {user_row}"
                )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": FORMAT_VERSION,
                "tokens": self.tokens,
                "offsets": self.offsets,
                "user_ids": self.user_ids,
                "target_pairs": self.target_pairs,
                "sid_hierarchy": self.sid_hierarchy,
                "sequence_length": self.sequence_length,
                "metadata": dict(self.metadata),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TargetPositionStore":
        payload = torch.load(Path(path), map_location="cpu")
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported target-position store version: {payload.get('format_version')}"
            )
        store = cls(
            tokens=payload["tokens"],
            offsets=payload["offsets"],
            user_ids=payload["user_ids"],
            target_pairs=payload["target_pairs"],
            sid_hierarchy=int(payload["sid_hierarchy"]),
            sequence_length=int(payload["sequence_length"]),
            metadata=payload.get("metadata", {}),
        )
        store.validate()
        return store


class TargetPositionDataset(Dataset):
    """Map-style dataset whose examples are `(user_row, target_position)` pairs."""

    def __init__(
        self,
        store: TargetPositionStore | str | Path,
        selected_indices: Optional[torch.Tensor | Sequence[int]] = None,
    ) -> None:
        self.store = (
            TargetPositionStore.load(store)
            if isinstance(store, (str, Path))
            else store
        )
        self.store.validate()
        if selected_indices is None:
            self.selected_indices = torch.arange(self.store.num_targets, dtype=torch.int64)
        else:
            self.selected_indices = torch.as_tensor(
                selected_indices, dtype=torch.int64
            ).flatten()
            if self.selected_indices.numel() and (
                int(self.selected_indices.min()) < 0
                or int(self.selected_indices.max()) >= self.store.num_targets
            ):
                raise IndexError("selected target index is outside the store")

    def __len__(self) -> int:
        return int(self.selected_indices.numel())

    def target_key(self, index: int) -> tuple[int, int]:
        pair_index = int(self.selected_indices[index])
        user_row, target_position = self.store.target_pairs[pair_index].tolist()
        return int(user_row), int(target_position)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_row, target_position = self.target_key(index)
        hierarchy = self.store.sid_hierarchy
        row_start = int(self.store.offsets[user_row])
        row_end = int(self.store.offsets[user_row + 1])
        row_tokens = self.store.tokens[row_start:row_end]

        target_end_item = target_position + 1
        start_item = max(0, target_end_item - self.store.max_items)
        sequence = row_tokens[start_item * hierarchy : target_end_item * hierarchy]
        return {
            "sequence_data": sequence.clone(),
            "user_id": self.store.user_ids[user_row].reshape(1).clone(),
        }
