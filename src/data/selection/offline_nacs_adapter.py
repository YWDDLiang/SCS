"""One stored candidate per scoring row; no second subsequence augmentation."""
from functools import partial
import hashlib
from pathlib import Path
import hydra
import torch
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader, Dataset
from src.data.loading.components.collate_functions import collate_fn_train
from src.data.selection.subsequence_candidate import SubsequenceCandidateDataset


class Int64CandidateRows(Dataset):
    def __init__(self, store, indices):
        self.dataset = SubsequenceCandidateDataset(store, indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {key: value.to(torch.int64) for key, value in self.dataset[index].items()}


def scoring_config(cfg):
    module = hydra.utils.instantiate(cfg.data_loading.datamodule)
    return module.stage_to_config[TrainerFn.FITTING]


def plain_collator(config):
    return partial(collate_fn_train, labels=config.labels, sequence_length=config.sequence_length,
                   masking_token=config.masking_token, padding_token=config.padding_token,
                   oov_token=config.get('oov_token', None))


def candidate_loader(cfg, store, indices, batch_size, num_workers, seed, pin_memory=True):
    config = scoring_config(cfg)
    if store.sequence_length != config.sequence_length:
        raise ValueError('Store/config sequence length mismatch')
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Int64CandidateRows(store, indices), batch_size=batch_size, shuffle=False,
                      drop_last=False, num_workers=num_workers, persistent_workers=num_workers > 0,
                      pin_memory=pin_memory, collate_fn=plain_collator(config), generator=generator)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def indices_hash(indices):
    return hashlib.sha256(indices.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()).hexdigest()
