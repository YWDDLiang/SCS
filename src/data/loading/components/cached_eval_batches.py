"""Trusted CPU evaluation batches, preserving saved order and batch boundaries."""
from functools import lru_cache
import copy
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset


@lru_cache(maxsize=None)
def _load_batches(path):
    # Only project-produced trusted cache files: serialized input/label dataclasses.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    batches = payload['batches'] if isinstance(payload, dict) else payload
    if not isinstance(batches, list):
        raise ValueError('Evaluation cache must contain a list of complete batches')
    if any(not isinstance(batch, (tuple, list)) or len(batch) != 2 for batch in batches):
        raise ValueError('Each cached evaluation batch must be an input/label pair')
    return batches


def _identity(batch):
    return batch


class CachedEvaluationBatches(Dataset):
    """One dataset item is one original evaluation batch, with normal finite EOF.

    Files load once per resolved path in each process. Returned batches are
    isolated copies, so evaluation cannot mutate the saved in-memory source.
    No TFRecord iterator or worker is created.
    """
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.batches = _load_batches(self.path)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return copy.deepcopy(self.batches[index])


def cached_evaluation_loader(path, seed, pin_memory=True):
    return DataLoader(CachedEvaluationBatches(path), batch_size=None, shuffle=False,
                      num_workers=0, pin_memory=pin_memory, collate_fn=_identity,
                      generator=torch.Generator().manual_seed(seed))
