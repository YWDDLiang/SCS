"""Same A1/A3 parents and slots; only copy/view action may change."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from lightning.pytorch.trainer.states import TrainerFn
from src.data.loading.datamodules.train_demand_allocation_datamodule import TrainDemandAllocationDataModule, file_sha256


class TrainActionChoiceDataModule(TrainDemandAllocationDataModule):
    def __init__(self,*args,action_assets=None,action_mode=None,action_choice_file=None,**kwargs):
        super().__init__(*args,**kwargs)
        if self.demand_mode!='random' or action_mode not in ('copy','policy'):
            raise ValueError('Action comparison must preserve the old random allocation')
        self.action_mode=action_mode; self.action_path=Path(action_assets).resolve()
        self.action_manifest_path=self.action_path/'manifest.json'
        self.action_identity=file_sha256(self.action_manifest_path)
        m=json.loads(self.action_manifest_path.read_text()); self.action_dataset=m['dataset']
        if not m['ready'] or m['source_manifest_sha256']!=self.demand_identity:
            raise RuntimeError('Action feature source mismatch')
        self.action_protocol_sha256=m['protocol_sha256']
        from scripts.train_choice_common import verify_protocol
        if verify_protocol()!=self.action_protocol_sha256: raise RuntimeError('Action protocol mismatch')
        for name in ('key_groups','key_other_support'):
            rec=m['files'][name+'.npy']; path=Path(rec['path'])
            if file_sha256(path)!=rec['sha256']: raise RuntimeError('Action feature identity changed')
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            setattr(self,name,value); self.asset_file_hashes[str(path)]=rec['sha256']
        if len(self.key_groups)!=len(self.key_records) or not np.array_equal(self.key_other_support,self.key_support-1):
            raise RuntimeError('Parent support does not exclude its own user')
        self.choice_sha256=None; self.choice_file=None
        self.actions=np.zeros(3,dtype=np.uint8)
        if action_mode=='policy':
            self.choice_file=Path(action_choice_file).resolve()
            self.choice_sha256=file_sha256(self.choice_file)
            choice=json.loads(self.choice_file.read_text())
            if choice['protocol_sha256']!=self.action_protocol_sha256 or choice['feature_manifest_sha256']!=self.action_identity:
                raise RuntimeError('Rule is not bound to this source/feature/protocol')
            values=choice['actions_by_group']
            if len(values)!=3 or any(type(v) is not int or v not in (0,1) for v in values):
                raise ValueError('Each group must use exactly copy or views')
            self.actions=np.asarray(values,dtype=np.uint8)
            self.asset_file_hashes[str(self.choice_file)]=self.choice_sha256
        self.asset_file_hashes[str(self.action_manifest_path)]=self.action_identity
        self.pool_sha256=hashlib.sha256(json.dumps(dict(parent_pool=self.pool_sha256,
            action_mode=action_mode, feature=self.action_identity, protocol=self.action_protocol_sha256,
            choice=self.choice_sha256),sort_keys=True).encode()).hexdigest()
        self.action_counts=np.zeros((3,2),dtype=np.int64)
        self.action_weights=np.zeros((3,2),dtype=np.float64)

    def batch_details_at(self,index):
        details=super().batch_details_at(index)
        keys=details['keys'][details['auxiliary']]
        details['action_groups']=np.asarray(self.key_groups[keys])
        details['use_view']=np.asarray(self.actions[details['action_groups']],dtype=bool)
        return details

    def collate_details(self,details):
        changed=details.copy()
        mask=details['auxiliary'].copy()
        mask[details['auxiliary']]=details['use_view']
        changed['auxiliary']=mask
        changed['view_ids']=details['view_ids'][details['use_view']]
        return super().collate_details(changed)

    def _account(self,details,output):
        super()._account(details,output)
        # Parent counts actual token changes correctly, but assumes every aux is a view.
        suppressed=int(self.view_novel_vs_source_pool[details['view_ids'][~details['use_view']]].sum())
        self.exposure['novel_inputs']-=suppressed
        self.exposure['novel_weight']-=suppressed/len(details['rows'])
        groups=details['action_groups']; actions=details['use_view'].astype(np.int64)
        np.add.at(self.action_counts,(groups,actions),1)
        np.add.at(self.action_weights,(groups,actions),1./len(details['rows']))

    def state_dict(self):
        state=super().state_dict()
        state.update(action_mode=self.action_mode,action_manifest_sha256=self.action_identity,
            action_protocol_sha256=self.action_protocol_sha256,action_choice_sha256=self.choice_sha256,
            action_assignment=self.actions.tolist(),action_counts=torch.from_numpy(self.action_counts.copy()),
            action_weights=torch.from_numpy(self.action_weights.copy()))
        return state

    def load_state_dict(self,state):
        if getattr(getattr(self.trainer,'state',None),'fn',None) in (TrainerFn.TESTING,TrainerFn.VALIDATING,TrainerFn.PREDICTING): return
        expected=dict(action_mode=self.action_mode,action_manifest_sha256=self.action_identity,
            action_protocol_sha256=self.action_protocol_sha256,action_choice_sha256=self.choice_sha256,
            action_assignment=self.actions.tolist())
        if any(state.get(k)!=v for k,v in expected.items()): raise RuntimeError('Saved action identity changed')
        super().load_state_dict(state)
        for key in ('action_counts','action_weights'):
            value=torch.as_tensor(state[key]).cpu().numpy()
            old=getattr(self,key)
            if value.shape!=old.shape or value.dtype!=old.dtype: raise RuntimeError('Action accounting shape/dtype changed')
            setattr(self,key,value.copy())
