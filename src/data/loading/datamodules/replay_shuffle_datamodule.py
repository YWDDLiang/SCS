"""Literal P1 occurrences, one immutable permutation, original cyclic K schedule."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from lightning.pytorch.trainer.states import TrainerFn
from src.data.loading.datamodules.ddp_scheduled_datamodule import DDPScheduledSequenceDataModule
from src.data.loading.components.fast_scheduled_collate import FastScheduledCollator
from src.data.selection.subsequence_candidate import SubsequenceCandidateStore


class _FixedShuffleBatches(IterableDataset):
    def __init__(self,module):self.module=module

    def __iter__(self):
        module=self.module;cursor=module.consumed_batches;rank=module.trainer.global_rank
        while True:
            # Same 128-global window/rank32 layout and metadata tuple as the
            # original _DDPTrainingBatches, with no finite epoch or short update.
            for index in range(cursor+rank,cursor+128,4):
                occurrences=module.occurrences_at(index)
                output=module._fast_collate(occurrences)
                yield (*output,{'ddp_global_batch_index':index})
            cursor+=128


class ReplayShuffleSequenceDataModule(DDPScheduledSequenceDataModule):
    def __init__(self,*args,assets_path=None,**kwargs):
        super().__init__(*args,**kwargs)
        if self.mode!='native_replay' or self.selection_ratio!=1. or self.selected_indices_path:
            raise ValueError('This module implements full fixed-shuffle replay only')
        if not self.fast_scheduled_collate or self.follow_source_trace or self.native_source_dir:
            raise ValueError('Full shuffle requires fast-data and an immutable original trace identity')
        path=assets_path or os.environ.get('REPLAY_SHUFFLE_ASSETS')
        if not path:raise ValueError('assets_path or REPLAY_SHUFFLE_ASSETS is required')
        self.assets_path=Path(path).resolve()
        self.asset_manifest=json.loads((self.assets_path/'manifest.json').read_text())
        if self.asset_manifest.get('ready') is not True or self.asset_manifest.get('seed')!=42:
            raise RuntimeError('Prepared ready=true seed42 asset manifest required')
        self.asset_file_hashes={}
        for name in ('occurrences.npy','permutation.npy','k_sizes.npy'):
            expected=self.asset_manifest['files'][name]['sha256']
            digest=hashlib.sha256()
            with (self.assets_path/name).open('rb') as stream:
                while block:=stream.read(8*1024*1024):digest.update(block)
            if digest.hexdigest()!=expected:raise RuntimeError(f'Shuffle asset SHA256 mismatch: {name}')
            self.asset_file_hashes[name]=expected
        self.asset_source_stream_id=self.asset_manifest['source_stream_id']
        self.pool_sha256=hashlib.sha256(json.dumps({'source_stream_id':self.asset_source_stream_id,
            'files':self.asset_file_hashes},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.occurrences=np.load(self.assets_path/'occurrences.npy',mmap_mode='r',allow_pickle=False)
        self.permutation=np.load(self.assets_path/'permutation.npy',mmap_mode='r',allow_pickle=False)
        self.k_sizes=np.load(self.assets_path/'k_sizes.npy',mmap_mode='r',allow_pickle=False)
        if self.occurrences.dtype!=np.int32 or self.occurrences.ndim!=2 or self.occurrences.shape[1]!=3:
            raise ValueError('Literal occurrence mmap must be int32[N,3]')
        if self.permutation.dtype!=np.int64 or self.permutation.ndim!=1 or len(self.permutation)!=len(self.occurrences):
            raise ValueError('Fixed permutation mmap must be int64[N]')
        if self.k_sizes.dtype!=np.int64 or self.k_sizes.ndim!=1 or not len(self.k_sizes) or np.any(self.k_sizes<=0):
            raise ValueError('K schedule must be nonempty positive int64[B]')
        self.k_offsets=np.r_[np.int64(0),np.cumsum(self.k_sizes,dtype=np.int64)]
        self.occurrence_count=len(self.occurrences);self.batch_period=len(self.k_sizes)
        if self.asset_manifest.get('total_occurrences')!=self.occurrence_count or self.asset_manifest.get('total_batches')!=self.batch_period:
            raise ValueError('Asset manifest count/shape mismatch')
        if not self.occurrence_count or int(self.k_offsets[-1])!=self.occurrence_count:
            raise ValueError('A complete K period must consume exactly the literal occurrence multiset')
        if self._source_identity()!=self.asset_source_stream_id:raise RuntimeError('Shuffle assets do not belong to supplied P1 trace UUID')
        self.last_occurrence_indices=None

    def occurrence_indices_at(self,global_batch_index):
        index=int(global_batch_index)
        if index<0:raise ValueError('Global batch index must be nonnegative')
        within=index%self.batch_period
        start=int(self.k_offsets[within]);end=int(self.k_offsets[within+1])
        # Since sum(K)==N, both fixed schedules wrap together. No RNG, new
        # permutation, membership filtering, candidate enumeration or dedup.
        return np.asarray(self.permutation[start:end],dtype=np.int64)

    def occurrences_at(self,global_batch_index):
        indices=self.occurrence_indices_at(global_batch_index)
        if np.any(indices<0) or np.any(indices>=self.occurrence_count):raise RuntimeError('Permutation contains an invalid literal occurrence identity')
        self.last_occurrence_indices=indices.copy()
        # Advanced indexing materializes only this microbatch, not the full mmap.
        values=np.asarray(self.occurrences[indices],dtype=np.int32).copy()
        return torch.from_numpy(values)

    def train_dataloader(self):
        self._require_world()
        if self.consumed_batches%128:raise RuntimeError('Shuffle replay resumes only at complete optimizer boundaries')
        self._identity_value=self._source_identity()
        if self._identity_value!=self.asset_source_stream_id:raise RuntimeError('Source UUID changed after shuffle assets were bound')
        config=self.stage_to_config[TrainerFn.FITTING]
        # Reuse the existing compact user-token store for reconstruction only.
        # No unique-candidate dataset or enumeration is introduced into this stream.
        self.store=SubsequenceCandidateStore.load(self.candidate_store_path)
        if self.store.sequence_length!=config.sequence_length:raise ValueError('Reconstruction sequence length mismatch')
        self._fast_collate=FastScheduledCollator(self.store,labels=config.labels,sequence_length=config.sequence_length,
            masking_token=config.masking_token,padding_token=config.padding_token,oov_token=config.get('oov_token',None))
        return (DataLoader(_FixedShuffleBatches(self),batch_size=None,num_workers=0,pin_memory=config.pin_memory,
            generator=torch.Generator().manual_seed(self.stream_seed)),)

    def state_dict(self):
        state=super().state_dict()
        state.update(replay_shuffle_format_version=1,replay_shuffle_asset_identity=self.pool_sha256,
            replay_shuffle_file_hashes=self.asset_file_hashes,literal_occurrence_count=self.occurrence_count,
            k_schedule_batch_period=self.batch_period,stream_kind='literal_P1_fixed_permutation_cycle')
        return state

    def load_state_dict(self,state):
        if getattr(getattr(self.trainer,'state',None),'fn',None) in (TrainerFn.TESTING,TrainerFn.VALIDATING,TrainerFn.PREDICTING):return
        if state.get('replay_shuffle_format_version')!=1 or state.get('replay_shuffle_asset_identity')!=self.pool_sha256:
            raise RuntimeError('Ordinary P1 checkpoint is not shuffled replay state, or shuffle assets differ')
        if state.get('replay_shuffle_file_hashes')!=self.asset_file_hashes or state.get('literal_occurrence_count')!=self.occurrence_count or state.get('k_schedule_batch_period')!=self.batch_period:
            raise RuntimeError('Shuffle checkpoint file identity/size mismatch')
        if self.restart_stream_on_restore:raise RuntimeError('Full shuffled replay continuation must restore its committed cursor, not restart it')
        if int(state.get('global_committed_cursor',-1))<0:raise RuntimeError('Invalid saved shuffle cursor')
        super().load_state_dict(state)
