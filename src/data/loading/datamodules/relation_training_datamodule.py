"""Approved replay-only repetition and same-event context interventions."""
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from lightning.pytorch.trainer.states import TrainerFn
from src.data.loading.components.cached_eval_batches import _identity
from src.data.loading.datamodules.replay_shuffle_datamodule import ReplayShuffleSequenceDataModule
from src.data.loading.datamodules.replay_selection_datamodule import _state_hash


class _LazyRelationBatches(IterableDataset):
    def __init__(self, module): self.module = module
    def __iter__(self):
        cursor = self.module.consumed_batches
        rank = self.module.trainer.global_rank
        while True:
            for index in range(cursor+rank, cursor+128, 4):
                yield {'relation_batch_index': index}
            cursor += 128


class RelationTrainingSequenceDataModule(ReplayShuffleSequenceDataModule):
    def __init__(self, *args, relation_arm=None, relation_assets=None, relation_output=None, **kwargs):
        super().__init__(*args, **kwargs)
        if relation_arm not in ('u2to4_x2', 'u5plus_x2', 'context'):
            raise ValueError('Unknown approved relation arm')
        self.relation_arm = relation_arm
        self.relation_path = Path(relation_assets).resolve()
        self.relation_output = Path(relation_output).resolve()
        path = self.relation_path/'manifest.json'
        self.relation_manifest = json.loads(path.read_text())
        self.relation_identity = hashlib.sha256(path.read_bytes()).hexdigest()
        m = self.relation_manifest
        self.boundary = 25600
        self.rotation = self.full_scan_position_at(self.boundary) % self.occurrence_count
        if (not m['ready'] or m['source_stream_id'] != self.asset_source_stream_id
                or m['total_occurrences'] != self.occurrence_count
                or m['base_rotation_occurrences'] != self.rotation
                or m['activation_microbatch'] != self.boundary):
            raise RuntimeError('Relation assets/source/activation mismatch')
        self.relation_file_hashes = {}
        def bind(record):
            file = Path(record['path'])
            h = hashlib.sha256()
            with file.open('rb') as f:
                for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
            if h.hexdigest() != record['sha256']:
                raise RuntimeError(f'Relation asset hash mismatch: {file}')
            self.relation_file_hashes[str(file)] = record['sha256']
            return np.load(file, mmap_mode='r', allow_pickle=False)
        self.occurrence_to_key = bind(m['context']['files']['occurrence_to_key.npy'])
        self.key_support = bind(m['files']['key_distinct_train_users.npy'])
        if relation_arm == 'context':
            self.replacement = bind(m['context']['files']['context_key_replacement.npy'])
            side = self.relation_path/'context_representatives.json'
            sm = json.loads(side.read_text())
            if not sm['ready'] or sm['source_stream_id'] != self.asset_source_stream_id:
                raise RuntimeError('Context representative identity mismatch')
            self.representatives_identity = hashlib.sha256(side.read_bytes()).hexdigest()
            self.representatives = bind(sm['files']['context_representative_occurrence_ids.npy'])
        else:
            group = m['groups'][relation_arm]
            self.copy_ids = bind(group['files']['copy_occurrence_ids.npy'])
            self.copy_positions = bind(group['files']['copy_output_positions.npy'])
            self.expanded_count = int(group['cycle_length'])
            if self.expanded_count != self.occurrence_count+len(self.copy_ids):
                raise RuntimeError('Expanded cycle count mismatch')
        self.intervention_active = False
        self.exposure = {'batches': 0, 'samples': 0, 'copies': 0, 'context_changes': 0,
            'support1': 0, 'support2to4': 0, 'support5plus': 0,
            'support1_weight': 0., 'support2to4_weight': 0., 'support5plus_weight': 0.}

    def full_scan_position_at(self, batch_cursor):
        cycles, within = divmod(int(batch_cursor), self.batch_period)
        return cycles*self.occurrence_count+int(self.k_offsets[within])

    def _activate(self):
        if self.intervention_active: return
        if int(self.trainer.global_step) != 200 or self.consumed_batches != self.boundary:
            raise RuntimeError('Intervention must first activate at committed update200')
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
        if torch.cuda.is_initialized(): rng['cuda_local'] = torch.cuda.get_rng_state(torch.cuda.current_device())
        record = {'rank': int(self.trainer.global_rank), 'global_step': 200,
            'committed_global_batch_cursor': self.consumed_batches,
            'model_sha256': _state_hash(self.trainer.lightning_module.state_dict()),
            'Adam_sha256': _state_hash(self.trainer.optimizers[0].state_dict()),
            'training_RNG_sha256': _state_hash(rng)}
        records = [None]*4
        if dist.is_available() and dist.is_initialized(): dist.all_gather_object(records, record)
        else: records = [record]
        if int(self.trainer.global_rank) == 0:
            self.relation_output.mkdir(parents=True, exist_ok=True)
            file = self.relation_output/'prefix200_audit.json'
            proof = {'global_step': 200, 'arm': self.relation_arm, 'ranks': records,
                     'source_stream_id': self.asset_source_stream_id, 'assets': self.relation_identity}
            if file.exists():
                if json.loads(file.read_text()) != proof: raise RuntimeError('Existing prefix audit differs')
            else: file.write_text(json.dumps(proof, indent=2))
        self.intervention_active = True

    def records_at(self, index):
        """Pure record lookup; activation and accounting occur only at transfer."""
        index = int(index)
        K = int(self.k_sizes[index % self.batch_period])
        copies = np.zeros(K, dtype=bool)
        if index < self.boundary or self.relation_arm == 'context':
            ids = super().occurrence_indices_at(index)
        else:
            start = self.full_scan_position_at(index)-self.full_scan_position_at(self.boundary)
            positions = (start+np.arange(K, dtype=np.int64)) % self.expanded_count
            before = np.searchsorted(self.copy_positions, positions, side='left')
            valid = before < len(self.copy_positions)
            copies[valid] = self.copy_positions[before[valid]] == positions[valid]
            base = positions-before
            ids = np.asarray(self.permutation[(self.rotation+base) % self.occurrence_count], dtype=np.int64).copy()
            ids[copies] = self.copy_ids[before[copies]]
        original_keys = np.asarray(self.occurrence_to_key[ids], dtype=np.int64)
        changed = np.zeros(K, dtype=bool)
        if index >= self.boundary and self.relation_arm == 'context':
            newkeys = self.replacement[original_keys]
            changed = newkeys != original_keys
            ids = ids.copy()
            ids[changed] = self.representatives[newkeys[changed]]
        rows = np.asarray(self.occurrences[ids], dtype=np.int32).copy()
        return rows, ids, copies, changed, original_keys

    def train_dataloader(self):
        super().train_dataloader()
        return (DataLoader(_LazyRelationBatches(self), batch_size=None, num_workers=0,
            collate_fn=_identity, pin_memory=False,
            generator=torch.Generator().manual_seed(self.stream_seed)),)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.training: return batch
        def consume(node):
            if isinstance(node, dict) and set(node) == {'relation_batch_index'}:
                index = int(node['relation_batch_index']); rank = int(self.trainer.global_rank)
                if index % 4 != rank or index-rank != self.consumed_batches:
                    raise RuntimeError('Relation rank cursor mismatch')
                if index >= self.boundary: self._activate()
                rows, ids, copies, changed, keys = self.records_at(index)
                self.last_occurrence_indices = ids.copy()
                output = self._fast_collate(torch.from_numpy(rows))
                e = self.exposure; e['batches'] += 1; e['samples'] += len(rows)
                e['copies'] += int(copies.sum()); e['context_changes'] += int(changed.sum())
                support = self.key_support[keys]
                for name, mask in [('support1', support == 1), ('support2to4', (support>=2)&(support<=4)), ('support5plus', support>=5)]:
                    n = int(mask.sum()); e[name] += n; e[name+'_weight'] += n/len(rows)
                return super(RelationTrainingSequenceDataModule, self).on_before_batch_transfer(
                    (*output, {'ddp_global_batch_index': index}), dataloader_idx)
            if not isinstance(node, (tuple, list)): return node
            children = [consume(x) for x in node]
            return tuple(children) if isinstance(node, tuple) else children
        return consume(batch)

    def state_dict(self):
        state = super().state_dict()
        scan = self.full_scan_position_at(self.consumed_batches)
        offset = max(0, scan-self.full_scan_position_at(self.boundary))
        state.update(relation_format_version=1, relation_arm=self.relation_arm,
            relation_asset_identity=self.relation_identity, relation_file_hashes=self.relation_file_hashes,
            relation_active=self.intervention_active, relation_expanded_offset=offset,
            relation_rotation=self.rotation, relation_exposure=dict(self.exposure))
        return state

    def load_state_dict(self, state):
        if getattr(getattr(self.trainer, 'state', None), 'fn', None) in (TrainerFn.TESTING, TrainerFn.VALIDATING, TrainerFn.PREDICTING): return
        if (state.get('relation_format_version') != 1 or state.get('relation_arm') != self.relation_arm
                or state.get('relation_asset_identity') != self.relation_identity
                or state.get('relation_file_hashes') != self.relation_file_hashes):
            raise RuntimeError('Relation checkpoint/assets mismatch')
        super().load_state_dict(state)
        self.intervention_active = bool(state['relation_active'])
        self.exposure = dict(state['relation_exposure'])
        if state['relation_expanded_offset'] != max(0, self.full_scan_position_at(self.consumed_batches)-self.full_scan_position_at(self.boundary)):
            raise RuntimeError('Relation checkpoint cursor mismatch')
        if self.consumed_batches > self.boundary and not self.intervention_active:
            raise RuntimeError('Post200 state lacks intervention identity')
