"""Four-rank sharding of one unchanged global native microbatch schedule."""
import copy
from functools import partial
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from lightning.pytorch.trainer.states import TrainerFn
from src.data.loading.datamodules.scheduled_datamodule import ScheduledSequenceDataModule
from src.data.loading.components.collate_functions import collate_fn_train
from src.data.loading.components.native_trace import NativeTraceReader, TraceExhaustedError, reconstruct_occurrences
from src.data.loading.components.native_source_stream import read_native_batches, NativeSourceUnavailable
from src.data.loading.components.cached_eval_batches import CachedEvaluationBatches, _identity
from src.data.selection.subsequence_candidate import SubsequenceCandidateStore, SubsequenceCandidateDataset, load_selected_candidate_indices


class RankCachedEvaluationBatches(Dataset):
    """Partition original finite batches without padding or repeated UIDs."""
    def __init__(self, path, rank, world_size=4):
        self.original = CachedEvaluationBatches(path)
        self.rank, self.world_size = rank, world_size
        self.indices = range(rank, len(self.original), world_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.original[self.indices[index]]


class _DDPTrainingBatches(IterableDataset):
    def __init__(self, module):
        self.module = module

    def __iter__(self):
        module = self.module
        cursor = module.consumed_batches
        rank, world = module.trainer.global_rank, module.trainer.world_size
        accumulation = module.global_accumulation_batches
        if module.mode == 'native_dynamic':
            schedule = read_native_batches(module.native_source_dir, start_batch=cursor)
        else:
            def trace_schedule():
                while True:
                    count = 0
                    reader = NativeTraceReader(module.source_trace_dir, follow=module.follow_source_trace)
                    module.trace_reader = reader
                    try:
                        for occurrence in reader.batches():
                            count += 1
                            module.trace_wait_seconds = reader.wait_seconds
                            yield occurrence
                    except TraceExhaustedError:
                        module.trace_wait_seconds = reader.wait_seconds
                        if module.mode != 'global_uniform' or not count:
                            raise
                        # Global training repeats its K schedule and candidate pool.
            schedule = trace_schedule()
        generator = torch.Generator().manual_seed(module.stream_seed)
        order, position = torch.empty(0, dtype=torch.int64), 0
        def global_indices(count):
            nonlocal order, position
            pieces = []
            while count:
                if module.selection_weights is not None:
                    # Keep the candidate pool unique while allowing a method
                    # to approximate repetition through weighted draws.
                    pieces.append(torch.multinomial(
                        module.selection_weights, count, replacement=True, generator=generator))
                    count = 0
                    continue
                if position == len(order):
                    order = torch.randperm(len(module.selected_indices), generator=generator)
                    position = 0
                take = min(count, len(order) - position)
                pieces.append(order[position:position + take])
                position += take
                count -= take
            return module.selected_indices[torch.cat(pieces)]
        try:
            if module.mode != 'native_dynamic':
                # Reproduce the shared global candidate RNG stream for every rank
                # before the saved boundary; no model-side RNG is consumed.
                for _ in range(cursor):
                    occurrence = next(schedule)
                    if module.mode == 'global_uniform':
                        global_indices(len(occurrence))
            while True:
                # Buffer one complete global optimizer window. A partial source
                # tail cannot create unequal rank steps or a short accumulation.
                window = [next(schedule) for _ in range(accumulation)]
                for local_index, record in enumerate(window):
                    index = cursor + local_index
                    if module.mode == 'global_uniform':
                        indices = global_indices(len(record))
                    if index % world != rank:
                        continue
                    if module.mode == 'native_dynamic':
                        output = record[:2]
                    elif module.mode == 'native_replay':
                        output = (module._fast_collate(record) if module.fast_scheduled_collate else
                                  module._ordinary_collate(reconstruct_occurrences(module.store, record, module.uid_to_row)))
                    elif module.fast_scheduled_collate:
                        output = module._fast_collate.collate_candidate_indices(indices)
                    else:
                        rows = [module.candidate_dataset[int(module.index_to_position[int(index)])] for index in indices]
                        for row in rows:
                            row['sequence_data'] = row['sequence_data'].to(torch.int64)
                        output = module._ordinary_collate(rows)
                    yield (*output, {'ddp_global_batch_index': index})
                cursor += accumulation
        except (TraceExhaustedError, NativeSourceUnavailable) as error:
            while True:
                yield {'native_trace_exhausted': str(error) + '; DDP source EOF is not convergence'}


class DDPScheduledSequenceDataModule(ScheduledSequenceDataModule):
    def __init__(self, *args, global_accumulation_batches=128, selection_weights_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if global_accumulation_batches != 128:
            raise ValueError('The approved DDP protocol requires 128 global microbatches per update')
        self.global_accumulation_batches = global_accumulation_batches
        self.selection_weights_path = selection_weights_path
        self.selection_weights = None
        self._identity_value = None
        self.pool_sha256 = None
        self._restored_pool_sha256 = None

    def _require_world(self):
        if self.trainer.world_size != 4:
            raise ValueError('DDPScheduledSequenceDataModule requires exactly four ranks')

    def _source_identity(self):
        trace_dir = Path(self.source_trace_dir or (Path(self.native_source_dir) / 'trace'))
        metadata = json.loads((trace_dir / 'metadata.json').read_text())
        identity = metadata.get('stream_id')
        if not identity:
            raise RuntimeError('DDP source trace must have a nonempty stream UUID')
        if self.native_source_dir:
            source = json.loads((Path(self.native_source_dir) / 'source.json').read_text())
            if source.get('stream_id') != identity:
                raise RuntimeError('Native source and trace UUID mismatch')
        if self._identity_value is not None and identity != self._identity_value:
            raise RuntimeError('DDP source UUID changed during the run')
        return identity

    def train_dataloader(self):
        self._require_world()
        if self.consumed_batches % self.global_accumulation_batches:
            raise RuntimeError('DDP resume cursor must be a complete 128-batch optimizer boundary')
        if self.mode == 'native_dynamic' and not self.native_source_dir:
            raise ValueError('Four-rank native training requires the external original CPU source')
        if not self.source_trace_dir and self.native_source_dir:
            self.source_trace_dir = str(Path(self.native_source_dir) / 'trace')
        self._identity_value = self._source_identity()
        config = self.stage_to_config[TrainerFn.FITTING]
        if self.mode != 'native_dynamic':
            self.store = SubsequenceCandidateStore.load(self.candidate_store_path)
            if self.store.sequence_length != config.sequence_length:
                raise ValueError('Candidate sequence length mismatch')
            self.uid_to_row = {int(uid): row for row, uid in enumerate(self.store.user_ids)}
            if len(self.uid_to_row) != self.store.num_rows:
                raise ValueError('Candidate user IDs must be unique')
            self.selected_indices = None
            if self.mode == 'global_uniform':
                self.selected_indices = (load_selected_candidate_indices(self.selected_indices_path) if self.selected_indices_path else
                    torch.randperm(self.store.num_candidates, generator=torch.Generator().manual_seed(self.stream_seed))[:max(1, round(self.store.num_candidates * self.selection_ratio))])
                self.candidate_dataset = SubsequenceCandidateDataset(self.store, self.selected_indices)
                if not len(self.candidate_dataset):
                    raise ValueError('Global selected pool is empty')
                if self.selection_weights_path:
                    weights_path = Path(self.selection_weights_path)
                    if weights_path.suffix == '.npy':
                        payload = torch.from_numpy(np.load(weights_path, mmap_mode='r').copy())
                    else:
                        payload = torch.load(weights_path, map_location='cpu', weights_only=True)
                    if isinstance(payload, dict):
                        payload = payload.get('weights', payload.get('sampling_weights'))
                    if payload is None:
                        raise ValueError('selection weights file must contain weights or sampling_weights')
                    weights = torch.as_tensor(payload, dtype=torch.float32).flatten()
                    if weights.numel() == self.store.num_candidates:
                        weights = weights[self.selected_indices]
                    if weights.numel() != self.selected_indices.numel():
                        raise ValueError('selection weights must align with the selected pool or full candidate store')
                    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
                        raise ValueError('selection weights must be finite and positive')
                    self.selection_weights = weights / weights.sum()
                self.pool_sha256 = hashlib.sha256(self.selected_indices.contiguous().numpy().tobytes()).hexdigest()
                if self._restored_pool_sha256 is not None and self._restored_pool_sha256 != self.pool_sha256:
                    raise RuntimeError("Restored DDP selected pool differs from current selection")
                if not self.fast_scheduled_collate:
                    self.index_to_position = {int(index): position for position, index in enumerate(self.selected_indices)}
            self._ordinary_collate = partial(collate_fn_train, labels=config.labels, sequence_length=config.sequence_length,
                masking_token=config.masking_token, padding_token=config.padding_token, oov_token=config.get('oov_token', None))
            if self.fast_scheduled_collate:
                from src.data.loading.components.fast_scheduled_collate import FastScheduledCollator
                self._fast_collate = FastScheduledCollator(self.store, labels=config.labels, sequence_length=config.sequence_length,
                    masking_token=config.masking_token, padding_token=config.padding_token, oov_token=config.get('oov_token', None))
        return (DataLoader(_DDPTrainingBatches(self), batch_size=None, num_workers=0,
            generator=torch.Generator().manual_seed(self.stream_seed), pin_memory=config.pin_memory),)

    def get_dataloader(self, stage):
        if stage == TrainerFn.FITTING:
            return self.train_dataloader()
        self._require_world()
        path = self.eval_cache_paths.get(stage.value)
        if not path:
            raise ValueError('DDP evaluation requires complete cached original batches')
        config = self.stage_to_config[stage]
        return (DataLoader(RankCachedEvaluationBatches(path, self.trainer.global_rank, self.trainer.world_size),
            batch_size=None, num_workers=0, pin_memory=config.pin_memory, collate_fn=_identity,
            generator=torch.Generator().manual_seed(self.stream_seed)),)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.training:
            return batch
        def consume(node):
            if isinstance(node, dict) and set(node) == {'native_trace_exhausted'}:
                raise TraceExhaustedError(node['native_trace_exhausted'])
            if not isinstance(node, (tuple, list)):
                return node
            if len(node) == 3 and isinstance(node[2], dict) and set(node[2]) == {'ddp_global_batch_index'}:
                index = int(node[2]['ddp_global_batch_index'])
                rank, world = self.trainer.global_rank, self.trainer.world_size
                if index % world != rank or index - rank != self.consumed_batches:
                    raise RuntimeError('DDP rank/global cursor order mismatch')
                # Every rank reports the same logical global group boundary.
                self.consumed_batches = index - rank + world
                return node[:2]
            children = [consume(child) for child in node]
            return tuple(children) if isinstance(node, tuple) else children
        return consume(batch)

    def state_dict(self):
        if self.consumed_batches % self.global_accumulation_batches:
            raise RuntimeError('Save DDP stream state only at a complete optimizer boundary')
        return {'format_version': 1, 'mode': self.mode, 'world_size': 4, 'global_accumulation_batches':128,
                'global_committed_cursor': self.consumed_batches, 'consumed_batches':self.consumed_batches,
                'source_stream_id':self._source_identity(), 'stream_seed':self.stream_seed,
                'selection_ratio':self.selection_ratio, 'pool_sha256':self.pool_sha256}

    def load_state_dict(self, state):
        if getattr(getattr(self.trainer, 'state', None), 'fn', None) in (TrainerFn.TESTING, TrainerFn.VALIDATING, TrainerFn.PREDICTING):
            return
        if state.get('world_size') != 4 or state.get('global_accumulation_batches') !=128 or state.get('mode') !=self.mode:
            raise RuntimeError('DDP checkpoint protocol mismatch')
        if state.get('source_stream_id') != self._source_identity():
            raise RuntimeError('DDP checkpoint source UUID mismatch')
        if state.get('stream_seed') != self.stream_seed:
            raise RuntimeError('DDP checkpoint stream seed mismatch')
        if not self.restart_stream_on_restore and state.get('selection_ratio') !=self.selection_ratio:
            raise RuntimeError('DDP checkpoint data selection configuration mismatch')
        self._restored_pool_sha256 = None if self.restart_stream_on_restore else state.get("pool_sha256")
        if not self.restart_stream_on_restore and self.pool_sha256 is not None and state.get('pool_sha256') !=self.pool_sha256:
            raise RuntimeError('DDP checkpoint global pool mismatch')
        cursor = int(state['global_committed_cursor'])
        if cursor %128:
            raise RuntimeError('DDP checkpoint contains an incomplete optimizer window')
        self.consumed_batches = 0 if self.restart_stream_on_restore else cursor
