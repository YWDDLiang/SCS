"""Original P2/P3 DDP stream with a raw, non-deduplicated candidate universe."""
from functools import partial
import hashlib,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.trainer.states import TrainerFn
from src.data.loading.datamodules.ddp_scheduled_datamodule import DDPScheduledSequenceDataModule,_DDPTrainingBatches
from src.data.loading.components.collate_functions import collate_fn_train
from src.data.loading.components.fast_scheduled_collate import FastScheduledCollator
from src.data.selection.raw_subsequence_candidate import RawSubsequenceCandidateStore
from src.data.selection.subsequence_candidate import SubsequenceCandidateDataset,load_selected_candidate_indices


class NoDedupScheduledDataModule(DDPScheduledSequenceDataModule):
    def __init__(self,*args,raw_manifest_path,**kwargs):
        super().__init__(*args,**kwargs)
        self.raw_manifest_path=Path(raw_manifest_path)
        self.raw_manifest=json.loads(self.raw_manifest_path.read_text())
        self.raw_store_sha256=hashlib.sha256(Path(self.candidate_store_path).read_bytes()).hexdigest()
        if self.raw_store_sha256!=self.raw_manifest['raw_store_sha256']:
            raise ValueError('Raw store checksum differs from preparation manifest')
        if self.mode!='global_uniform' or not self.fast_scheduled_collate:
            raise ValueError('No-dedup experiment requires global_uniform and verified fast-data')

    def train_dataloader(self):
        self._require_world()
        if self.consumed_batches%128:raise ValueError('Restore requires a complete optimizer window')
        self._identity_value=self._source_identity()
        config=self.stage_to_config[TrainerFn.FITTING]
        self.store=RawSubsequenceCandidateStore.load(self.candidate_store_path)
        if self.store.sequence_length!=config.sequence_length:raise ValueError('Sequence length mismatch')
        self.uid_to_row={int(uid):row for row,uid in enumerate(self.store.user_ids)}
        self.selected_indices=(load_selected_candidate_indices(self.selected_indices_path) if self.selected_indices_path else
            torch.randperm(self.store.num_candidates,generator=torch.Generator().manual_seed(self.stream_seed))[:max(1,round(self.store.num_candidates*self.selection_ratio))])
        if len(self.selected_indices)!=round(self.store.num_candidates*self.selection_ratio):
            raise ValueError('Selected raw-entry count does not match frozen retention')
        # Raw IDs are unique; their truncated inputs need not be unique.
        self.candidate_dataset=SubsequenceCandidateDataset(self.store,self.selected_indices)
        self.pool_sha256=hashlib.sha256(self.selected_indices.contiguous().numpy().tobytes()).hexdigest()
        if self._restored_pool_sha256 is not None and self._restored_pool_sha256!=self.pool_sha256:
            raise ValueError('Restored raw selected pool differs')
        self._ordinary_collate=partial(collate_fn_train,labels=config.labels,sequence_length=config.sequence_length,
            masking_token=config.masking_token,padding_token=config.padding_token,oov_token=config.get('oov_token',None))
        self._fast_collate=FastScheduledCollator(self.store,labels=config.labels,sequence_length=config.sequence_length,
            masking_token=config.masking_token,padding_token=config.padding_token,oov_token=config.get('oov_token',None))
        return (DataLoader(_DDPTrainingBatches(self),batch_size=None,num_workers=0,
            generator=torch.Generator().manual_seed(self.stream_seed),pin_memory=config.pin_memory),)

    def state_dict(self):
        state=super().state_dict()
        state.update(pool_kind='raw_subsequence_no_dedup',raw_store_sha256=self.raw_store_sha256,
            raw_candidate_count=self.raw_manifest['raw_count'],canonical_candidate_count=self.raw_manifest['canonical_count'])
        return state

    def load_state_dict(self,state):
        if getattr(getattr(self.trainer,'state',None),'fn',None) in (TrainerFn.TESTING,TrainerFn.VALIDATING,TrainerFn.PREDICTING):return
        if state.get('pool_kind')!='raw_subsequence_no_dedup' or state.get('raw_store_sha256')!=self.raw_store_sha256:
            raise ValueError('Checkpoint raw-pool identity mismatch')
        super().load_state_dict(state)
