from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch

from src.data.loading.components.collate_functions import collate_fn_train
from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)


def collate_fn_train_with_user_ids(
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    labels: Dict[str, callable],  # type: ignore
    sequence_length: int = 200,
    masking_token: int = 1,
    padding_token: int = 0,
    oov_token: Optional[int] = None,
    data_augmentation_functions: Optional[List[Dict[str, callable]]] = None,  # type: ignore
    user_id_field: str = "user_id",
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """Preserve validation user IDs while reusing the native train collator."""
    if isinstance(batch, list):
        if not batch or any(user_id_field not in row for row in batch):
            raise KeyError(f"missing validation user ID field: {user_id_field}")
        user_ids = torch.cat(
            [torch.as_tensor(row[user_id_field]).flatten() for row in batch]
        )
    else:
        if user_id_field not in batch:
            raise KeyError(f"missing validation user ID field: {user_id_field}")
        user_ids = torch.as_tensor(batch[user_id_field]).flatten()

    model_input, label_data = collate_fn_train(
        batch,
        labels,
        sequence_length=sequence_length,
        masking_token=masking_token,
        padding_token=padding_token,
        oov_token=oov_token,
        data_augmentation_functions=data_augmentation_functions,
    )
    model_input.user_id_list = user_ids
    return model_input, label_data
