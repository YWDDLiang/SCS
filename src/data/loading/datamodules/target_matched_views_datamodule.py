"""Six approved target-matched controls and ordered views on fixed extra slots."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.trainer.states import TrainerFn

from src.data.loading.components.interfaces import SequentialModelInputData, SequentialModuleLabelData
from src.data.loading.datamodules.relation_training_datamodule import RelationTrainingSequenceDataModule


class TargetMatchedViewsDataModule(RelationTrainingSequenceDataModule):
    def __init__(self, *args, target_view_variant=None, target_view_assets=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.relation_arm not in ('u2to4_x2', 'u5plus_x2'):
            raise ValueError('Only approved dispersed extra-slot parents are supported')
        if target_view_variant not in ('matched', 'views'):
            raise ValueError('Unknown target/history intervention')
        self.target_view_variant = target_view_variant
        self.target_view_path = Path(target_view_assets).resolve()
        name = 'matched_manifest.json' if target_view_variant == 'matched' else 'manifest.json'
        self.target_view_manifest_path = self.target_view_path/name
        self.target_view_identity = hashlib.sha256(self.target_view_manifest_path.read_bytes()).hexdigest()
        self.target_view_manifest = json.loads(self.target_view_manifest_path.read_text())
        m = self.target_view_manifest
        parent_hash = m.get('parent_relation_manifest_sha256', m.get('relation_manifest_sha256'))
        parent_arm = m.get('source_arm', m.get('parent_group'))
        if (not m['ready'] or m['source_stream_id'] != self.asset_source_stream_id
                or parent_hash != self.relation_identity or parent_arm != self.relation_arm
                or m['copy_count'] != len(self.copy_ids) or m['cycle_length'] != self.expanded_count
                or m['activation_microbatch'] != self.boundary):
            raise RuntimeError('New assets do not match the immutable extra-slot/source identity')
        self.target_view_file_hashes = {}

        def bind(name):
            record = m['files'][name]
            file = Path(record['path'])
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != record['sha256']:
                raise RuntimeError(f'New asset hash mismatch: {file}')
            self.target_view_file_hashes[str(file)] = record['sha256']
            value = np.load(file, mmap_mode='r', allow_pickle=False)
            if len(value) != len(self.copy_ids):
                raise RuntimeError('Every extra-slot ordinal must have one frozen asset row')
            return value

        if target_view_variant == 'matched':
            self.matched_ids = bind('matched_occurrence_ids.npy')
            self.matched_flags = bind('matched_flags.npy')
            if self.matched_ids.dtype != np.int64 or self.matched_flags.dtype != np.uint8:
                raise ValueError('Unexpected matched-asset dtype')
        else:
            if self.relation_arm != 'u2to4_x2':
                raise ValueError('New views are approved only for the common 2-4 support group')
            self.view_positions = bind('view_history_positions.npy')
            self.view_changed = bind('view_changed.npy')
            self.view_novel = bind('view_novel_vs_original_pool.npy')
            if self.view_positions.dtype != np.uint16 or self.view_positions.shape[1:] != (29,):
                raise ValueError('Expected uint16[M,29] TRAIN history positions')
        # "copies" now counts unchanged extra inputs; auxiliary_slots counts all
        # original copy slots, whether replaced or generated. Preserve both.
        self.exposure.update(auxiliary_slots=0, matched_changed_keys=0,
            matched_changed_inputs=0, matched_changed_relations=0,
            matched_changed_users=0, matched_eligible_other_B=0,
            generated_changed=0, generated_novel_in_source=0)

    def batch_details_at(self, index):
        rows, ids, auxiliary, _, parent_keys = super().records_at(index)
        ordinal = np.empty(0, dtype=np.int64)
        flags = np.zeros(len(rows), dtype=np.uint8)
        changed = np.zeros(len(rows), dtype=bool)
        novel = np.zeros(len(rows), dtype=bool)
        if auxiliary.any():
            start = self.full_scan_position_at(index)-self.full_scan_position_at(self.boundary)
            positions = (start+np.arange(len(rows), dtype=np.int64)) % self.expanded_count
            ordinal = np.searchsorted(self.copy_positions, positions[auxiliary], side='left')
            if not np.array_equal(self.copy_positions[ordinal], positions[auxiliary]):
                raise RuntimeError('Extra-slot ordinal lookup differs from the parent stream')
            if self.target_view_variant == 'matched':
                ids = ids.copy()
                ids[auxiliary] = self.matched_ids[ordinal]
                rows = np.asarray(self.occurrences[ids], dtype=np.int32).copy()
                flags[auxiliary] = self.matched_flags[ordinal]
                changed = (flags & 2) != 0
            else:
                changed[auxiliary] = self.view_changed[ordinal]
                novel[auxiliary] = self.view_novel[ordinal]
        keys = np.asarray(self.occurrence_to_key[ids], dtype=np.int64)
        return dict(rows=rows, ids=ids, auxiliary=auxiliary, ordinal=ordinal,
            changed=changed, novel=novel, flags=flags, keys=keys, parent_keys=parent_keys)

    def records_at(self, index):
        d = self.batch_details_at(index)
        return d['rows'], d['ids'], d['auxiliary'], d['changed'], d['keys']

    def _collate_view_batch(self, d):
        if not d['changed'].any():
            return self._fast_collate(torch.from_numpy(d['rows']))
        collator = self._fast_collate
        occurrences = torch.from_numpy(d['rows']).to(torch.int64)
        uids, starts, targets = occurrences.unbind(1)
        locations = torch.searchsorted(collator.sorted_uids, uids.contiguous())
        if torch.any(locations >= len(collator.sorted_uids)) or not torch.equal(collator.sorted_uids[locations], uids):
            raise KeyError('View source user absent from original TRAIN store')
        rows = collator.uid_rows[locations]
        hierarchy = self.store.sid_hierarchy
        if hierarchy != 4 or collator.sequence_length != 120:
            raise ValueError('This approved view asset uses four SID tokens and 29 history items')
        lengths = (targets-starts).clamp(max=29)
        effective_starts = targets-lengths
        item_positions = effective_starts[:, None]+torch.arange(30)[None, :]
        extra_rows = np.flatnonzero(d['auxiliary'])
        changed_extra = d['changed'][extra_rows]
        changed_rows = torch.from_numpy(extra_rows[changed_extra])
        frozen = torch.from_numpy(np.asarray(self.view_positions[d['ordinal'][changed_extra]], dtype=np.int64).copy())
        valid = torch.arange(29)[None, :] < lengths[changed_rows, None]
        current = item_positions[changed_rows, :29]
        item_positions[changed_rows, :29] = torch.where(valid, frozen, current)
        # Keep the true target at the same position as the original label transform.
        item_positions[torch.arange(len(rows)), lengths] = targets
        indices = self.store.offsets[rows, None, None]+item_positions[:, :, None]*hierarchy+torch.arange(hierarchy)[None, None, :]
        indices = indices.reshape(len(rows), 120).clamp(max=self.store.tokens.numel()-1)
        sequences = self.store.tokens[indices].to(torch.int64)
        positions = collator.positions[None, :]
        sequences.masked_fill_(positions >= ((lengths+1)*hierarchy)[:, None], collator.padding_token)
        user_ids = torch.full_like(sequences, collator.padding_token)
        user_ids[:, 0] = uids
        model_input, model_labels = SequentialModelInputData(), SequentialModuleLabelData()
        for field, values in (('sequence_data', sequences), ('user_id', user_ids)):
            if field in collator.labels:
                transformed = collator.labels[field].transform.transform_label(
                    sequence=values, padding_token=collator.padding_token, masking_token=collator.masking_token)
                model_input.transformed_sequences[field] = transformed.sequence
                model_labels.labels[field] = transformed.labels
                model_labels.label_location[field] = transformed.label_location
                model_labels.attention_mask[field] = transformed.attention_mask
            else:
                model_input.transformed_sequences[field] = values
            if model_input.mask is None:
                model_input.mask = (values != collator.padding_token).long()
        return model_input, model_labels

    def collate_details(self, details):
        if self.target_view_variant == 'views':
            return self._collate_view_batch(details)
        return self._fast_collate(torch.from_numpy(details['rows']))

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.training:
            return batch

        def consume(node):
            if isinstance(node, dict) and set(node) == {'relation_batch_index'}:
                index, rank = int(node['relation_batch_index']), int(self.trainer.global_rank)
                if index % 4 != rank or index-rank != self.consumed_batches:
                    raise RuntimeError('New intervention rank cursor mismatch')
                if index >= self.boundary:
                    self._activate()
                d = self.batch_details_at(index)
                self.last_occurrence_indices = d['ids'].copy()
                output = self.collate_details(d)
                e = self.exposure
                e['batches'] += 1
                e['samples'] += len(d['rows'])
                e['auxiliary_slots'] += int(d['auxiliary'].sum())
                e['copies'] += int((d['auxiliary'] & ~d['changed']).sum())
                e['context_changes'] += int(d['changed'].sum())
                if self.target_view_variant == 'matched':
                    for bit, name in enumerate(('keys', 'inputs', 'relations', 'users')):
                        e['matched_changed_'+name] += int(((d['flags'] & (1 << bit)) != 0).sum())
                    e['matched_eligible_other_B'] += int(((d['flags'] & 16) != 0).sum())
                else:
                    e['generated_changed'] += int(d['changed'].sum())
                    e['generated_novel_in_source'] += int(d['novel'].sum())
                support = self.key_support[d['keys']]
                for name, mask in (('support1', support == 1), ('support2to4', (support>=2)&(support<=4)), ('support5plus', support>=5)):
                    n = int(mask.sum())
                    e[name] += n
                    e[name+'_weight'] += n/len(d['rows'])
                # Skip the parent relation hook; its accounting assumes literal copies.
                return super(RelationTrainingSequenceDataModule, self).on_before_batch_transfer(
                    (*output, {'ddp_global_batch_index': index}), dataloader_idx)
            if not isinstance(node, (tuple, list)):
                return node
            children = [consume(x) for x in node]
            return tuple(children) if isinstance(node, tuple) else children
        return consume(batch)

    def state_dict(self):
        state = super().state_dict()
        state.update(target_view_format_version=1, target_view_variant=self.target_view_variant,
            target_view_asset_identity=self.target_view_identity,
            target_view_manifest_path=str(self.target_view_manifest_path),
            target_view_file_hashes=self.target_view_file_hashes,
            target_view_exposure=dict(self.exposure))
        return state

    def load_state_dict(self, state):
        if getattr(getattr(self.trainer, 'state', None), 'fn', None) in (TrainerFn.TESTING, TrainerFn.VALIDATING, TrainerFn.PREDICTING):
            return
        if (state.get('target_view_format_version') != 1
                or state.get('target_view_variant') != self.target_view_variant
                or state.get('target_view_asset_identity') != self.target_view_identity
                or state.get('target_view_file_hashes') != self.target_view_file_hashes):
            raise RuntimeError('New intervention checkpoint/assets mismatch')
        super().load_state_dict(state)
        if state['target_view_exposure'] != self.exposure:
            raise RuntimeError('Saved new intervention exposure counters disagree')
