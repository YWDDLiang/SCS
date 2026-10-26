"""Paired copy/view action on the exact same clean-source >=5 extra slots."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from lightning.pytorch.trainer.states import TrainerFn
from src.data.loading.datamodules.train_demand_allocation_datamodule import TrainDemandAllocationDataModule,file_sha256

class TrainActionDataModule(TrainDemandAllocationDataModule):
    def __init__(self,*args,action_assets=None,action_mode=None,**kwargs):
        # Parent binds audited source, store, K and evaluation. None of its
        # q/10%-allocation arrays are used by the overridden stream below.
        super().__init__(*args,**kwargs)
        if action_mode not in ('copy','views'):raise ValueError('Expected paired copy/views')
        self.action_mode=action_mode
        self.demand_mode=action_mode
        self.action_path=Path(action_assets).resolve()
        self.action_identity=file_sha256(self.action_path/'manifest.json')
        m=json.loads((self.action_path/'manifest.json').read_text())
        if (not m['ready'] or m['base_manifest_sha256']!=self.demand_identity
                or m['source_stream_id']!=self.asset_source_stream_id
                or m['source_occurrences']!=self.occurrence_count or m['activation_microbatch']!=self.boundary):
            raise RuntimeError('Action/source binding mismatch')
        def bind(name):
            r=m['files'][name];p=Path(r['path'])
            if file_sha256(p)!=r['sha256']:raise RuntimeError('Action asset hash mismatch')
            self.asset_file_hashes[str(p.resolve())]=r['sha256']
            a=np.load(p,mmap_mode='r',allow_pickle=False)
            if str(a.dtype)!=r['dtype'] or list(a.shape)!=r['shape']:raise RuntimeError('Action shape mismatch')
            return a
        self.copy_source_positions=bind('copy_source_positions.npy')
        self.copy_output_positions=bind('copy_output_positions.npy')
        self.view_positions=bind('view_positions.npy')
        self.action_view_changed=bind('view_changed.npy')
        self.view_novel_vs_source_pool=bind('view_novel.npy')
        self.aux_count=len(self.copy_source_positions)
        self.expanded_count=self.occurrence_count+self.aux_count
        self.rotation=self.full_scan_position_at(self.boundary)%self.occurrence_count
        if self.expanded_count!=m['cycle_length'] or self.rotation!=m['rotation']:
            raise RuntimeError('Action cycle mismatch')
        if action_mode=='copy':
            self.view_novel_vs_source_pool=np.zeros(self.aux_count,dtype=np.bool_)
        self.pool_sha256=hashlib.sha256(json.dumps(dict(base=self.pool_sha256,
            action=self.action_identity,mode=action_mode),sort_keys=True).encode()).hexdigest()

    def cursors_at(self,batch_cursor):
        scan=self.full_scan_position_at(batch_cursor)
        anchor=self.full_scan_position_at(self.boundary)
        post=max(0,scan-anchor)
        cycles,within=divmod(post,self.expanded_count)
        aux=cycles*self.aux_count+int(np.searchsorted(self.copy_output_positions,within,side='left'))
        return dict(scan_position=scan,post200_positions=post,auxiliary_ordinal=aux,
            base_reader_position=scan-aux)

    def batch_details_at(self,index):
        size=int(self.k_sizes[index%self.batch_period])
        scan=self.full_scan_position_at(index)
        if index<self.boundary:
            auxiliary=np.zeros(size,dtype=bool)
            positions=(scan+np.arange(size))%self.occurrence_count
            ordinal=np.empty(0,dtype=np.int64)
        else:
            pp=(scan-self.full_scan_position_at(self.boundary)+np.arange(size))%self.expanded_count
            before=np.searchsorted(self.copy_output_positions,pp,side='left')
            auxiliary=(before<self.aux_count)&(self.copy_output_positions[np.minimum(before,self.aux_count-1)]==pp)
            positions=(self.rotation+pp-before)%self.occurrence_count
            ordinal=before[auxiliary]
            positions[auxiliary]=self.copy_source_positions[ordinal]
        ids=np.asarray(self.source_order[positions],dtype=np.int64)
        keys=np.asarray(self.source_key_order[positions],dtype=np.int64)
        rows=np.asarray(self.occurrences[ids],dtype=np.int32).copy()
        if auxiliary.any() and not np.all(self.key_support[keys[auxiliary]]>=5):
            raise RuntimeError('Action auxiliary must be source-supported >=5')
        return dict(rows=rows,ids=ids,keys=keys,auxiliary=auxiliary,ordinal=ordinal,
            view_ids=ordinal,source_positions=positions,lengths=np.asarray(self.key_lengths[keys]),
            targets=np.asarray(self.key_targets[keys]))

    def collate_details(self,details):
        if self.action_mode=='copy':return self._fast_collate(torch.from_numpy(details['rows']))
        return super().collate_details(details)

    def state_dict(self):
        state=super().state_dict()
        state.update(action_mode=self.action_mode,action_manifest_sha256=self.action_identity)
        return state

    def load_state_dict(self,state):
        if getattr(getattr(self.trainer,'state',None),'fn',None) not in (TrainerFn.TESTING,TrainerFn.VALIDATING,TrainerFn.PREDICTING):
            if state.get('action_mode')!=self.action_mode or state.get('action_manifest_sha256')!=self.action_identity:
                raise RuntimeError('Action checkpoint/manifest mismatch')
        return super().load_state_dict(state)
