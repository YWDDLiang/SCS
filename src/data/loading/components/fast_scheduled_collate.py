"""CPU vectorized collation of stored occurrences; preserves ordinary label transforms."""
import torch
from src.data.loading.components.interfaces import SequentialModelInputData, SequentialModuleLabelData


class FastScheduledCollator:
    def __init__(self, store, labels, sequence_length, masking_token, padding_token, oov_token=None):
        if oov_token is not None:
            raise ValueError('Fast scheduled collation requires oov_token=None')
        if set(labels) - {'sequence_data'}:
            raise ValueError('Fast scheduled collation only supports sequence_data labels')
        if store.tokens.device.type != 'cpu':
            raise ValueError('Fast scheduled collation is a CPU data path')
        self.store = store
        self.labels = labels
        self.sequence_length = sequence_length
        self.masking_token, self.padding_token = masking_token, padding_token
        self.sorted_uids, self.uid_rows = torch.sort(store.user_ids.to(torch.int64))
        if torch.any(self.sorted_uids[1:] == self.sorted_uids[:-1]):
            raise ValueError('User IDs must be unique')
        self.positions = torch.arange(sequence_length, dtype=torch.int64)

    def __call__(self, occurrences):
        occurrences = occurrences.to(torch.int64)
        uids = occurrences[:, 0].contiguous()
        locations = torch.searchsorted(self.sorted_uids, uids)
        if torch.any(locations >= len(self.sorted_uids)):
            raise KeyError('Occurrence user ID is absent from store')
        if not torch.equal(self.sorted_uids[locations], uids):
            raise KeyError('Occurrence user ID is absent from store')
        return self._collate(self.uid_rows[locations], occurrences[:, 1], occurrences[:, 2], uids)

    def collate_candidate_indices(self, indices):
        triples = self.store.candidate_triples[torch.as_tensor(indices, dtype=torch.int64)].to(torch.int64)
        rows, starts, targets = triples.unbind(1)
        return self._collate(rows, starts, targets, self.store.user_ids[rows])

    def _collate(self, rows, starts, targets, uids):
        hierarchy = self.store.sid_hierarchy
        raw_lengths = (targets + 1 - starts) * hierarchy
        lengths = raw_lengths.clamp(max=self.sequence_length)
        shifts = (raw_lengths - self.sequence_length).clamp(min=0)
        token_starts = self.store.offsets[rows] + starts * hierarchy + shifts
        positions = self.positions.unsqueeze(0)
        token_indices = token_starts.unsqueeze(1) + positions
        # Padding positions may extend past the final store row; their gathered
        # values are discarded, so clamp only those irrelevant indices safely.
        sequences = self.store.tokens[token_indices.clamp(max=self.store.tokens.numel() - 1)].to(torch.int64)
        sequences.masked_fill_(positions >= lengths.unsqueeze(1), self.padding_token)
        user_ids = torch.full((len(rows), self.sequence_length), self.padding_token, dtype=torch.int64)
        user_ids[:, 0] = uids
        model_input = SequentialModelInputData()
        model_labels = SequentialModuleLabelData()
        for field, values in (('sequence_data', sequences), ('user_id', user_ids)):
            if field in self.labels:
                transformed = self.labels[field].transform.transform_label(
                    sequence=values, padding_token=self.padding_token, masking_token=self.masking_token)
                model_input.transformed_sequences[field] = transformed.sequence
                model_labels.labels[field] = transformed.labels
                model_labels.label_location[field] = transformed.label_location
                model_labels.attention_mask[field] = transformed.attention_mask
            else:
                model_input.transformed_sequences[field] = values
            if model_input.mask is None:
                model_input.mask = (values != self.padding_token).long()
        return model_input, model_labels
