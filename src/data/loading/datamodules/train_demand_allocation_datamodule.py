"""Clean TRAIN replay and frozen 10% ordered views, with allocation as sole arm difference."""
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from lightning.pytorch.trainer.states import TrainerFn

from src.data.loading.components.cached_eval_batches import _identity
from src.data.loading.components.fast_scheduled_collate import FastScheduledCollator
from src.data.loading.components.interfaces import SequentialModelInputData, SequentialModuleLabelData
from src.data.loading.datamodules.ddp_scheduled_datamodule import DDPScheduledSequenceDataModule
from src.data.loading.datamodules.replay_selection_datamodule import _state_hash
from src.data.selection.subsequence_candidate import SubsequenceCandidateStore


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class _LazyDemandBatches(IterableDataset):
    def __init__(self, module):
        self.module = module

    def __iter__(self):
        cursor, rank = self.module.consumed_batches, int(self.module.trainer.global_rank)
        while True:
            for index in range(cursor + rank, cursor + 128, 4):
                yield {'demand_batch_index': index}
            cursor += 128


class TrainDemandAllocationDataModule(DDPScheduledSequenceDataModule):
    boundary = 25600

    def __init__(self, *args, demand_assets=None, demand_mode=None, demand_output=None, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.mode != 'native_replay' or self.selection_ratio != 1. or self.selected_indices_path
                or not self.fast_scheduled_collate or self.follow_source_trace or self.native_source_dir
                or self.restart_stream_on_restore or self.stream_seed != 42):
            raise ValueError('Demand allocation requires immutable full source replay and seed42')
        if demand_mode not in ('random', 'matched'):
            raise ValueError('Demand mode must be random or matched')
        self.demand_mode = demand_mode
        self.demand_path = Path(demand_assets).resolve()
        self.demand_output = Path(demand_output).resolve()
        self.demand_manifest_path = self.demand_path / 'manifest.json'
        self.demand_identity = file_sha256(self.demand_manifest_path)
        self.demand_manifest = json.loads(self.demand_manifest_path.read_text())
        manifest = self.demand_manifest
        if manifest.get('ready') is not True:
            raise RuntimeError('Demand assets must be ready=true before loading')
        self.asset_source_stream_id = manifest['source_stream_id']
        self.original_path = Path(manifest['original_assets']).resolve()
        self.original_manifest_identity = file_sha256(self.original_path / 'manifest.json')
        if self.original_manifest_identity != manifest['source_manifest_sha256']:
            raise RuntimeError('Original P1 manifest identity differs from clean-source provenance')
        original = json.loads((self.original_path / 'manifest.json').read_text())
        if original.get('source_stream_id') != self.asset_source_stream_id or original.get('ready') is not True:
            raise RuntimeError('Original P1 ready/UUID mismatch')
        self.asset_file_hashes = {}

        def bind(record, base, array=True):
            path = Path(record['path'])
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
            digest = file_sha256(path)
            if digest != record['sha256']:
                raise RuntimeError(f'Demand asset SHA256 mismatch: {path}')
            self.asset_file_hashes[str(path)] = digest
            if not array:
                return path
            value = np.load(path, mmap_mode='r', allow_pickle=False)
            if str(value.dtype) != record['dtype'] or list(value.shape) != record['shape']:
                raise RuntimeError(f'Demand asset dtype/shape mismatch: {path}')
            return value

        for name in ('occurrences.npy', 'permutation.npy', 'k_sizes.npy'):
            record = manifest['original_source_files'][name]
            if record['sha256'] != original['files'][name]['sha256']:
                raise RuntimeError('Original P1 file provenance mismatch: ' + name)
            value = bind(record, self.original_path)
            if name != 'permutation.npy':
                setattr(self, name.removesuffix('.npy'), value)
        for name in ('source_order', 'source_key_order', 'key_records', 'key_counts', 'key_targets',
                'key_lengths', 'key_support', 'view_offsets', 'view_positions', 'view_key_ids',
                'view_novel_vs_source_pool',
                'aux_lengths', 'aux_random_view_ids', 'aux_demand_view_ids', 'source_lengths',
                'dev_target_counts', 'dev_uids', 'audit_uids'):
            setattr(self, name, bind(manifest['files'][name + '.npy'], self.demand_path))
        self.source_store_path = bind(manifest['files']['source_store.pt'], self.demand_path, array=False)
        if Path(self.candidate_store_path).resolve() != self.source_store_path:
            raise RuntimeError('Only the physically cropped clean-source store may reconstruct training')
        self.eval_manifest_path = self.demand_path / 'evaluation' / 'eval_manifest.json'
        self.eval_identity = file_sha256(self.eval_manifest_path)
        self.eval_manifest = json.loads(self.eval_manifest_path.read_text())
        if self.eval_manifest.get('ready') is not True:
            raise RuntimeError('Internal evaluation assets must be ready=true')
        if self.eval_manifest.get('source_stream_id', self.asset_source_stream_id) != self.asset_source_stream_id:
            raise RuntimeError('Internal evaluation source UUID mismatch')
        if self.eval_manifest.get('source_manifest_sha256', self.demand_identity) != self.demand_identity:
            raise RuntimeError('Internal evaluation must bind the frozen clean-source manifest')
        if self.eval_manifest['source']['sha256'] != manifest['train_only_input']['sha256']:
            raise RuntimeError('Internal evaluation and clean source must use the same TRAIN-only input')
        for split, uids in (('internal_dev', self.dev_uids), ('internal_audit', self.audit_uids)):
            uid_hash = hashlib.sha256(np.asarray(uids, dtype=np.int64).tobytes()).hexdigest()
            if uid_hash != self.eval_manifest['splits'][split]['membership_uid_sha256']:
                raise RuntimeError('Internal evaluation UID membership differs from the frozen allocation split')
        self.internal_eval_paths = {}
        for split in ('internal_dev', 'internal_audit'):
            path = bind(self.eval_manifest['files'][split + '.pt'], self.eval_manifest_path.parent, array=False)
            self.internal_eval_paths[split] = str(path)
        if Path(self.eval_cache_paths.get('validate', '')).resolve() != Path(self.internal_eval_paths['internal_dev']):
            raise RuntimeError('Early stopping must use only the frozen internal dev cache')
        self.official_test_path = str(Path(self.eval_cache_paths['test']).resolve())
        self.official_test_identity = file_sha256(self.official_test_path)
        self.asset_file_hashes[self.official_test_path] = self.official_test_identity
        self.pool_sha256 = hashlib.sha256(json.dumps(dict(
            source_stream_id=self.asset_source_stream_id, original_manifest_sha256=self.original_manifest_identity,
            demand_manifest_sha256=self.demand_identity, eval_manifest_sha256=self.eval_identity,
            mode=self.demand_mode, files=self.asset_file_hashes), sort_keys=True).encode()).hexdigest()
        self.occurrence_count = len(self.source_order)
        self.batch_period = len(self.k_sizes)
        if not self.occurrence_count or not self.batch_period or np.any(self.k_sizes <= 0):
            raise ValueError('Clean source and positive K schedule must be nonempty')
        self.k_offsets = np.r_[np.int64(0), np.cumsum(self.k_sizes, dtype=np.int64)]
        self.k_period_samples = int(self.k_offsets[-1])
        self.aux_count = len(self.aux_lengths)
        if self.aux_count != (self.occurrence_count + 8) // 9:
            raise ValueError('Frozen auxiliary cycle must have ceil(source occurrences / 9) slots')
        if (self.source_order.dtype != np.int64 or self.source_key_order.dtype != np.int32
                or len(self.source_key_order) != self.occurrence_count or self.key_records.shape[1:] != (3,)
                or self.k_sizes.dtype != np.int64 or self.view_positions.shape[1:] != (29,)
                or self.view_positions.dtype != np.uint16):
            raise ValueError('Unexpected demand source/view/K format')
        self.aux_view_ids = self.aux_random_view_ids if demand_mode == 'random' else self.aux_demand_view_ids
        if len(self.aux_view_ids) != self.aux_count or self.aux_view_ids.dtype != np.int64:
            raise ValueError('Frozen allocation cycle shape/type mismatch')
        if self._source_identity() != self.asset_source_stream_id:
            raise RuntimeError('Demand source does not match supplied original trace UUID')
        self._identity_value = self.asset_source_stream_id
        self.intervention_active = False
        self.last_occurrence_indices = None
        self.exposure = dict(batches=0, samples=0, base_slots=0, auxiliary_slots=0,
            changed_inputs=0, novel_inputs=0, base_weight=0., auxiliary_weight=0., changed_weight=0., novel_weight=0.)
        # The first dimension is base / auxiliary. Every occurrence contributes
        # one count and 1/K to each frozen key, target, length and support group.
        self.exposure_arrays = {}
        for name, size in (('key', len(self.key_records)), ('target', len(self.dev_target_counts)),
                ('length', 30), ('support', 4)):
            self.exposure_arrays[name + '_count'] = np.zeros((2, size), dtype=np.int64)
            self.exposure_arrays[name + '_weight'] = np.zeros((2, size), dtype=np.float64)

    def full_scan_position_at(self, batch_cursor):
        cycles, within = divmod(int(batch_cursor), self.batch_period)
        if batch_cursor < 0:
            raise ValueError('Batch cursor must be nonnegative')
        return cycles * self.k_period_samples + int(self.k_offsets[within])

    def cursors_at(self, batch_cursor):
        scan = self.full_scan_position_at(batch_cursor)
        anchor = self.full_scan_position_at(self.boundary)
        post = max(0, scan - anchor)
        aux = post // 10
        return dict(scan_position=scan, post200_positions=post,
            auxiliary_ordinal=aux, base_reader_position=scan - aux)

    def batch_details_at(self, index):
        """Pure frozen-array lookup, independent of global RNG and training state."""
        index = int(index)
        size = int(self.k_sizes[index % self.batch_period])
        absolute = self.full_scan_position_at(index) + np.arange(size, dtype=np.int64)
        if index < self.boundary:
            auxiliary = np.zeros(size, dtype=bool)
            source_positions = absolute % self.occurrence_count
            ordinal = np.empty(0, dtype=np.int64)
        else:
            anchor = self.full_scan_position_at(self.boundary)
            post = absolute - anchor
            auxiliary = post % 10 == 9
            # floor(j/10) counts preceding aux slots at a base position j.
            source_positions = (anchor + post - post // 10) % self.occurrence_count
            ordinal = (post[auxiliary] // 10) % self.aux_count
        ids = np.asarray(self.source_order[source_positions], dtype=np.int64).copy()
        keys = np.asarray(self.source_key_order[source_positions], dtype=np.int64).copy()
        rows = np.asarray(self.occurrences[ids], dtype=np.int32).copy()
        view_ids = np.asarray(self.aux_view_ids[ordinal], dtype=np.int64)
        if auxiliary.any():
            keys[auxiliary] = self.view_key_ids[view_ids]
            rows[auxiliary] = self.key_records[keys[auxiliary]]
            ids[auxiliary] = -1  # generated views have no literal occurrence ID
            if not np.array_equal(self.key_lengths[keys[auxiliary]], self.aux_lengths[ordinal]):
                raise RuntimeError('Frozen allocation changed a paired auxiliary length')
        return dict(rows=rows, ids=ids, keys=keys, auxiliary=auxiliary, ordinal=ordinal,
            view_ids=view_ids, source_positions=source_positions,
            lengths=np.asarray(self.key_lengths[keys], dtype=np.int64),
            targets=np.asarray(self.key_targets[keys], dtype=np.int64))

    def collate_details(self, details):
        if not details['auxiliary'].any():
            return self._fast_collate(torch.from_numpy(details['rows']))
        collator = self._fast_collate
        occurrences = torch.from_numpy(details['rows']).long()
        uids, starts, targets = occurrences.unbind(1)
        locations = torch.searchsorted(collator.sorted_uids, uids.contiguous())
        if torch.any(locations >= len(collator.sorted_uids)) or not torch.equal(collator.sorted_uids[locations], uids):
            raise KeyError('Ordered-view source user is absent from the clean store')
        rows = collator.uid_rows[locations]
        hierarchy = self.store.sid_hierarchy
        if hierarchy != 4 or collator.sequence_length != 120:
            raise ValueError('Approved views require hierarchy4 and sequence_length120')
        lengths = (targets - starts).clamp(max=29)
        positions = targets[:, None] - lengths[:, None] + torch.arange(30)[None, :]
        aux_rows = torch.from_numpy(np.flatnonzero(details['auxiliary']))
        frozen = torch.from_numpy(np.asarray(self.view_positions[details['view_ids']], dtype=np.int64).copy())
        valid = torch.arange(29)[None, :] < lengths[aux_rows, None]
        positions[aux_rows, :29] = torch.where(valid, frozen, positions[aux_rows, :29])
        positions[torch.arange(len(rows)), lengths] = targets
        indices = self.store.offsets[rows, None, None] + positions[:, :, None] * hierarchy + torch.arange(hierarchy)[None, None, :]
        indices = indices.reshape(len(rows), 120).clamp(max=self.store.tokens.numel() - 1)
        sequences = self.store.tokens[indices].long()
        sequences.masked_fill_(collator.positions[None, :] >= ((lengths + 1) * hierarchy)[:, None], collator.padding_token)
        user_ids = torch.full_like(sequences, collator.padding_token)
        user_ids[:, 0] = uids
        model_input, model_labels = SequentialModelInputData(), SequentialModuleLabelData()
        for field, values in (('sequence_data', sequences), ('user_id', user_ids)):
            if field in collator.labels:
                transformed = collator.labels[field].transform.transform_label(sequence=values,
                    padding_token=collator.padding_token, masking_token=collator.masking_token)
                model_input.transformed_sequences[field] = transformed.sequence
                model_labels.labels[field] = transformed.labels
                model_labels.label_location[field] = transformed.label_location
                model_labels.attention_mask[field] = transformed.attention_mask
            else:
                model_input.transformed_sequences[field] = values
            if model_input.mask is None:
                model_input.mask = (values != collator.padding_token).long()
        return model_input, model_labels

    def train_dataloader(self):
        self._require_world()
        if self.consumed_batches % 128:
            raise RuntimeError('Demand continuation requires complete global128 update boundary')
        if self._source_identity() != self.asset_source_stream_id:
            raise RuntimeError('Original source UUID changed after clean assets were bound')
        config = self.stage_to_config[TrainerFn.FITTING]
        self.store = SubsequenceCandidateStore.load(self.source_store_path)
        if self.store.sequence_length != config.sequence_length:
            raise ValueError('Clean store reconstruction sequence length mismatch')
        if (torch.unique(self.store.user_ids).numel() != self.store.num_rows
                or not torch.equal(torch.sort(self.store.user_ids.long()).values, torch.arange(self.store.num_rows))):
            raise ValueError('Clean source store must preserve every original UID exactly once')
        lengths = (self.store.offsets[1:] - self.store.offsets[:-1]) // self.store.sid_hierarchy
        if not np.array_equal(lengths.numpy(), self.source_lengths[self.store.user_ids.long().numpy()]):
            raise ValueError('Clean store must physically remove every hidden TRAIN last item')
        self._fast_collate = FastScheduledCollator(self.store, labels=config.labels,
            sequence_length=config.sequence_length, masking_token=config.masking_token,
            padding_token=config.padding_token, oov_token=config.get('oov_token', None))
        return (DataLoader(_LazyDemandBatches(self), batch_size=None, num_workers=0, collate_fn=_identity,
            pin_memory=False, generator=torch.Generator().manual_seed(self.stream_seed)),)

    def _activate(self):
        if self.intervention_active:
            return
        if int(self.trainer.global_step) != 200 or self.consumed_batches != self.boundary:
            raise RuntimeError('First auxiliary batch must follow committed update200')
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
        if torch.cuda.is_initialized():
            rng['cuda_local'] = torch.cuda.get_rng_state(torch.cuda.current_device())
        record = dict(rank=int(self.trainer.global_rank), global_step=200,
            committed_global_batch_cursor=self.consumed_batches,
            model_sha256=_state_hash(self.trainer.lightning_module.state_dict()),
            Adam_sha256=_state_hash(self.trainer.optimizers[0].state_dict()),
            training_RNG_sha256=_state_hash(rng))
        records = [None] * 4
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(records, record)
        else:
            records = [record]
        if int(self.trainer.global_rank) == 0:
            self.demand_output.mkdir(parents=True, exist_ok=True)
            path = self.demand_output / 'prefix200_audit.json'
            proof = dict(global_step=200, mode=self.demand_mode, ranks=records,
                source_stream_id=self.asset_source_stream_id, original_manifest_sha256=self.original_manifest_identity,
                source_manifest_sha256=self.demand_identity, eval_manifest_sha256=self.eval_identity,
                scope='Hashes only; no extra step200 weights; DDP model resume is not certified')
            if path.exists() and json.loads(path.read_text()) != proof:
                raise RuntimeError('Existing prefix200 proof differs; preserve it')
            if not path.exists():
                path.write_text(json.dumps(proof, indent=2))
        self.intervention_active = True

    def _account(self, details, output):
        size = len(details['rows'])
        aux = details['auxiliary']
        e = self.exposure
        e['batches'] += 1
        e['samples'] += size
        e['auxiliary_slots'] += int(aux.sum())
        e['base_slots'] += int((~aux).sum())
        e['base_weight'] += float((~aux).sum()) / size
        e['auxiliary_weight'] += float(aux.sum()) / size
        if aux.any():
            # Different positions can have identical SID tokens for repeated
            # items. Count actual model-input change against the parent suffix.
            ordinary = self._fast_collate(torch.from_numpy(details['rows'][aux]))
            changed = torch.any(output[0].transformed_sequences['sequence_data'][aux]
                != ordinary[0].transformed_sequences['sequence_data'], dim=1)
            e['changed_inputs'] += int(changed.sum())
            e['changed_weight'] += float(changed.sum()) / size
            novel = int(self.view_novel_vs_source_pool[details['view_ids']].sum())
            e['novel_inputs'] += novel
            e['novel_weight'] += novel / size
        support = np.asarray(self.key_support[details['keys']])
        groups = np.where(support == 0, 0, np.where(support == 1, 1, np.where(support <= 4, 2, 3)))
        fields = dict(key=details['keys'], target=details['targets'], length=details['lengths'], support=groups)
        for which, mask in enumerate((~aux, aux)):
            for name, values in fields.items():
                np.add.at(self.exposure_arrays[name + '_count'][which], values[mask], 1)
                np.add.at(self.exposure_arrays[name + '_weight'][which], values[mask], 1. / size)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.training:
            return batch

        def consume(node):
            if isinstance(node, dict) and set(node) == {'demand_batch_index'}:
                index, rank = int(node['demand_batch_index']), int(self.trainer.global_rank)
                if index % 4 != rank or index - rank != self.consumed_batches:
                    raise RuntimeError('Demand rank/global cursor mismatch')
                if index >= self.boundary:
                    self._activate()
                details = self.batch_details_at(index)
                self.last_occurrence_indices = details['ids'].copy()
                output = self.collate_details(details)
                self._account(details, output)
                return super(TrainDemandAllocationDataModule, self).on_before_batch_transfer(
                    (*output, {'ddp_global_batch_index': index}), dataloader_idx)
            if not isinstance(node, (tuple, list)):
                return node
            children = [consume(child) for child in node]
            return tuple(children) if isinstance(node, tuple) else children
        return consume(batch)

    def state_dict(self):
        state = super().state_dict()
        state.update(demand_format_version=1, demand_mode=self.demand_mode,
            demand_manifest_sha256=self.demand_identity, original_manifest_sha256=self.original_manifest_identity,
            eval_manifest_sha256=self.eval_identity, asset_file_hashes=dict(self.asset_file_hashes),
            demand_active=self.intervention_active, demand_cursors=self.cursors_at(self.consumed_batches),
            demand_exposure=dict(self.exposure),
            demand_exposure_arrays={name: torch.from_numpy(value.copy()) for name, value in self.exposure_arrays.items()})
        return state

    def load_state_dict(self, state):
        if getattr(getattr(self.trainer, 'state', None), 'fn', None) in (TrainerFn.TESTING, TrainerFn.VALIDATING, TrainerFn.PREDICTING):
            return
        expected = dict(demand_format_version=1, demand_mode=self.demand_mode,
            demand_manifest_sha256=self.demand_identity, original_manifest_sha256=self.original_manifest_identity,
            eval_manifest_sha256=self.eval_identity, asset_file_hashes=self.asset_file_hashes)
        if any(state.get(name) != value for name, value in expected.items()):
            raise RuntimeError('Demand checkpoint source/view/allocation/evaluation identity mismatch')
        if int(state.get('global_committed_cursor', -1)) < 0:
            raise RuntimeError('Invalid saved demand cursor')
        super().load_state_dict(state)
        if state.get('demand_cursors') != self.cursors_at(self.consumed_batches):
            raise RuntimeError('Demand saved reader/aux cursor mismatch')
        self.intervention_active = bool(state['demand_active'])
        if self.consumed_batches > self.boundary and not self.intervention_active:
            raise RuntimeError('Post200 checkpoint lacks intervention identity')
        self.exposure = copy.deepcopy(state['demand_exposure'])
        arrays = state['demand_exposure_arrays']
        if set(arrays) != set(self.exposure_arrays):
            raise RuntimeError('Demand exposure state lacks complete count/1K arrays')
        for name, value in arrays.items():
            array = torch.as_tensor(value).cpu().numpy()
            if array.shape != self.exposure_arrays[name].shape or array.dtype != self.exposure_arrays[name].dtype:
                raise RuntimeError('Demand exposure state shape/dtype mismatch')
            self.exposure_arrays[name] = array.copy()

    def select_test_split(self, split):
        if split not in ('official_test', 'internal_audit'):
            raise ValueError('Only the two frozen terminal evaluation splits are supported')
        path = self.official_test_path if split == 'official_test' else self.internal_eval_paths[split]
        self.eval_cache_paths['test'] = path
