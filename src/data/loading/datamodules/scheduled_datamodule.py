"""Native occurrence recording, exact replay, and global shuffled training streams."""
from functools import partial
import json
from pathlib import Path
import math
import torch
from torch.utils.data import DataLoader, IterableDataset
from lightning.pytorch.trainer.states import TrainerFn
from src.data.loading.datamodules.sequence_datamodule import SequenceDataModule
from src.data.loading.components.collate_functions import collate_fn_train
from src.data.loading.components.native_trace import NativeTraceReader, NativeTraceWriter, TraceExhaustedError, reconstruct_occurrences
from src.data.selection.subsequence_candidate import SubsequenceCandidateStore, SubsequenceCandidateDataset, load_selected_candidate_indices

class _OriginalSourceBatches(IterableDataset):
    def __init__(self, module):
        self.module = module

    def __iter__(self):
        from src.data.loading.components.native_source_stream import read_native_batches, NativeSourceUnavailable
        try:
            yield from read_native_batches(self.module.native_source_dir, self.module.consumed_batches)
        except NativeSourceUnavailable as error:
            # Defer EOF until consumption, preserving Lightning's lookahead.
            while True:
                yield {'native_trace_exhausted': str(error)}

class _ScheduledBatches(IterableDataset):
    def __init__(self, module):
        self.module = module

    def __iter__(self):
        module = self.module
        store = module.store
        uid_to_row = {int(uid): row for row, uid in enumerate(store.user_ids)}
        generator = torch.Generator().manual_seed(module.stream_seed)
        dataset = SubsequenceCandidateDataset(store, module.selected_indices)
        order, position = torch.empty(0, dtype=torch.int64), 0
        def schedule():
            while True:
                count = 0
                try:
                    reader = NativeTraceReader(module.source_trace_dir, follow=module.follow_source_trace)
                    module.trace_reader = reader
                    for occurrence in reader.batches():
                        module.trace_wait_seconds = reader.wait_seconds
                        count += 1
                        yield occurrence
                except TraceExhaustedError as error:
                    module.trace_wait_seconds = reader.wait_seconds
                    if module.mode == "native_replay" or not count:
                        # Lightning looks ahead before returning the last real
                        # batch. Defer the explicit EOF failure until consumption.
                        while True:
                            yield {"native_trace_exhausted": str(error)}
                    # Global candidates form an infinite stream; repeat only the
                    # reference K schedule, never stop at a selected-pool boundary.
        for index, occurrences in enumerate(schedule()):
            if isinstance(occurrences, dict) and "native_trace_exhausted" in occurrences:
                yield occurrences
                continue
            if module.mode == 'native_replay':
                if index < module.consumed_batches:
                    continue
                if module.fast_scheduled_collate:
                    output = module._fast_collate(occurrences)
                else:
                    rows = reconstruct_occurrences(store, occurrences, uid_to_row)
                    output = module._ordinary_collate(rows)
            else:
                pieces, remaining = [], len(occurrences)
                while remaining:
                    if position == len(order):
                        order = torch.randperm(len(dataset), generator=generator)
                        position = 0
                    take = min(remaining, len(order) - position)
                    pieces.append(order[position:position + take])
                    position += take
                    remaining -= take
                if index < module.consumed_batches:
                    continue
                positions = torch.cat(pieces)
                if module.fast_scheduled_collate:
                    output = module._fast_collate.collate_candidate_indices(module.selected_indices[positions])
                else:
                    rows = [dataset[int(position)] for position in positions]
                    for row in rows:
                        row['sequence_data'] = row['sequence_data'].to(torch.int64)
                    output = module._ordinary_collate(rows)
            yield (*output, {'scheduled_batch_index': index})

class ScheduledSequenceDataModule(SequenceDataModule):
    def __init__(self, train_dataloader_config=None, val_dataloader_config=None, test_dataloader_config=None, predict_dataloader_config=None, mode='native_dynamic', candidate_store_path=None, trace_dir=None, source_trace_dir=None, selected_indices_path=None, selection_ratio=1.0, stream_seed=42, trace_chunk_batches=128, restart_stream_on_restore=False, follow_source_trace=False, fast_scheduled_collate=False, native_source_dir=None, eval_cache_paths=None, loader_context=None):
        super().__init__(train_dataloader_config, val_dataloader_config, test_dataloader_config, predict_dataloader_config)
        if mode not in ('native_dynamic', 'native_replay', 'global_uniform'):
            raise ValueError(f'Unknown scheduled mode: {mode}')
        if not 0 < selection_ratio <= 1:
            raise ValueError('selection_ratio must be in (0, 1]')
        if follow_source_trace and mode != "native_replay":
            raise ValueError("follow_source_trace is only supported for native_replay")
        self.follow_source_trace = follow_source_trace
        self.fast_scheduled_collate = fast_scheduled_collate
        self.trace_wait_seconds = 0.0
        self.mode, self.candidate_store_path = mode, candidate_store_path
        self.trace_dir, self.source_trace_dir = trace_dir, source_trace_dir
        self.selected_indices_path, self.selection_ratio = selected_indices_path, selection_ratio
        self.stream_seed, self.trace_chunk_batches = stream_seed, trace_chunk_batches
        self.restart_stream_on_restore = restart_stream_on_restore
        self.consumed_batches = 0
        self.writer = None
        self._loader_generators = {}
        self.native_source_dir = native_source_dir
        self.eval_cache_paths = eval_cache_paths or {}
        self.loader_context = loader_context

    def get_dataloader(self, stage):
        if stage != TrainerFn.FITTING and self.eval_cache_paths.get(stage.value):
            from src.data.loading.components.cached_eval_batches import cached_evaluation_loader
            config = self.stage_to_config[stage]
            return (cached_evaluation_loader(self.eval_cache_paths[stage.value], self.stream_seed, config.pin_memory),)
        # The original base constructs the loader, but generator assignment occurs
        # before iteration and thus before any worker seed is drawn.
        loaders = super().get_dataloader(stage)
        for loader in loaders:
            generator = self._loader_generators.setdefault(stage, torch.Generator().manual_seed(self.stream_seed))
            loader.generator = generator
            if self.loader_context and loader.num_workers:
                loader.multiprocessing_context = self.loader_context
        return loaders

    def _get_partial_collate_fn(self, config):
        result = super()._get_partial_collate_fn(config)
        if self.mode == 'native_dynamic' and config is self.stage_to_config[TrainerFn.FITTING]:
            result = partial(result, return_occurrences=True)
        return result

    def train_dataloader(self):
        if self.trainer.world_size != 1:
            raise ValueError('ScheduledSequenceDataModule currently requires single-rank training')
        config = self.stage_to_config[TrainerFn.FITTING]
        if self.mode == 'native_dynamic':
            if self.native_source_dir:
                return (DataLoader(_OriginalSourceBatches(self), batch_size=None, num_workers=0,
                    generator=torch.Generator().manual_seed(self.stream_seed), pin_memory=config.pin_memory),)
            if config.num_workers == 0:
                raise ValueError('native_dynamic requires num_workers > 0 to isolate collator RNG from model dropout')
            if not self.trace_dir:
                raise ValueError('native_dynamic requires trace_dir')
            self.writer = NativeTraceWriter(self.trace_dir, {'candidate_store_path': self.candidate_store_path, 'stream_seed': self.stream_seed, 'rank': 0, 'data_folder': config.data_folder, 'sequence_length': config.sequence_length, 'dataset_config': str(config.dataset_config), 'collate_fn': str(config.collate_fn)}, self.trace_chunk_batches)
            return self.get_dataloader(TrainerFn.FITTING)
        if not self.source_trace_dir or not self.candidate_store_path:
            raise ValueError('replay/global require source_trace_dir and candidate_store_path')
        self.store = SubsequenceCandidateStore.load(self.candidate_store_path)
        if self.store.sequence_length != config.sequence_length:
            raise ValueError('candidate store sequence_length differs from training config')
        if torch.unique(self.store.user_ids).numel() != self.store.num_rows:
            raise ValueError('occurrence reconstruction requires unique user_ids')
        self.selected_indices = None
        if self.mode == 'global_uniform':
            if self.selected_indices_path:
                self.selected_indices = load_selected_candidate_indices(self.selected_indices_path)
            else:
                count = max(1, round(self.store.num_candidates * self.selection_ratio))
                self.selected_indices = torch.randperm(self.store.num_candidates, generator=torch.Generator().manual_seed(self.stream_seed))[:count]
            if not len(self.selected_indices):
                raise ValueError('global selected pool must be non-empty')
        self._ordinary_collate = partial(collate_fn_train, labels=config.labels, sequence_length=config.sequence_length, masking_token=config.masking_token, padding_token=config.padding_token, oov_token=config.get('oov_token', None))
        if self.fast_scheduled_collate:
            from src.data.loading.components.fast_scheduled_collate import FastScheduledCollator
            self._fast_collate = FastScheduledCollator(self.store, labels=config.labels,
                sequence_length=config.sequence_length, masking_token=config.masking_token,
                padding_token=config.padding_token, oov_token=config.get('oov_token', None))
        # One main-process iterable preserves batch order and avoids prefetch cursor
        # errors. The collator is deterministic for the TIGER causal label transform.
        return (DataLoader(_ScheduledBatches(self), batch_size=None, num_workers=0, generator=torch.Generator().manual_seed(self.stream_seed), pin_memory=config.pin_memory),)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not getattr(self.trainer, 'training', False):
            return batch

        def consume(node):
            if isinstance(node, dict) and set(node) == {"native_trace_exhausted"}:
                raise TraceExhaustedError(node["native_trace_exhausted"])
            if not isinstance(node, (tuple, list)):
                return node
            if len(node) == 3:
                trace = node[2]
                native_trace = (self.mode == 'native_dynamic' and torch.is_tensor(trace)
                                and trace.dtype == torch.int32 and trace.ndim == 2
                                and trace.shape[1] == 3)
                scheduled_trace = (self.mode != 'native_dynamic' and isinstance(trace, dict)
                                   and set(trace) == {'scheduled_batch_index'})
                if native_trace:
                    if self.writer is not None:
                        self.writer.append(trace)
                    self.consumed_batches += 1
                    return node[:2]
                if scheduled_trace:
                    self.consumed_batches = int(trace['scheduled_batch_index']) + 1
                    return node[:2]
            # Lightning preserves the tuple of train loaders as a CombinedLoader
            # envelope. Strip only the inner training trace, preserving that envelope.
            children = [consume(child) for child in node]
            return tuple(children) if isinstance(node, tuple) else children

        return consume(batch)

    def state_dict(self):
        if self.writer:
            self.writer.flush()
        if self.mode == 'native_dynamic' and self.consumed_batches:
            if self.native_source_dir:
                info = json.loads((Path(self.native_source_dir) / 'source.json').read_text())
                return {'consumed_batches': self.consumed_batches, 'mode': self.mode,
                    'native_exact_resume_supported': True, 'native_source_stream_id': info['stream_id'],
                    'native_source_dir': str(self.native_source_dir)}
            # Native worker RNG/prefetch state is not checkpointed by the base loader.
            # Do not claim an exact native continuation after checkpoint restart.
            return {'consumed_batches': self.consumed_batches, 'mode': self.mode, 'native_exact_resume_supported': False}
        if self.mode == 'native_replay':
            metadata = json.loads((Path(self.source_trace_dir) / 'metadata.json').read_text())
            return {'consumed_batches': self.consumed_batches, 'mode': self.mode,
                'source_trace_stream_id': metadata.get('stream_id')}
        return {'consumed_batches': self.consumed_batches, 'mode': self.mode}

    def load_state_dict(self, state_dict):
        trainer_fn = getattr(getattr(self.trainer, "state", None), "fn", None)
        if trainer_fn in (TrainerFn.TESTING, TrainerFn.VALIDATING, TrainerFn.PREDICTING):
            return
        if state_dict.get('mode') != self.mode:
            raise ValueError('scheduled stream checkpoint mode mismatch')
        if self.mode == 'native_replay':
            metadata = json.loads((Path(self.source_trace_dir) / 'metadata.json').read_text())
            current_id = metadata.get('stream_id')
            saved_id = state_dict.get('source_trace_stream_id')
            if current_id != saved_id:
                raise RuntimeError('Replay checkpoint source trace UUID mismatch or missing identity; only both-legacy traces without UUID remain compatible')
        self.consumed_batches = 0 if self.restart_stream_on_restore else int(state_dict.get('consumed_batches', 0))
        if self.mode == 'native_dynamic' and self.consumed_batches:
            if not self.native_source_dir or not state_dict.get('native_exact_resume_supported'):
                raise RuntimeError('Legacy native checkpoint lacks worker RNG/prefetch state; exact continuation is unavailable')
            info = json.loads((Path(self.native_source_dir) / 'source.json').read_text())
            if state_dict.get('native_source_stream_id') != info['stream_id']:
                raise RuntimeError('Native checkpoint belongs to a different source stream')

    def restart_data_stream(self):
        self.consumed_batches = 0

    def teardown(self, stage=None):
        if self.writer:
            self.writer.flush(complete=bool(getattr(self.trainer, "should_stop", False)))
