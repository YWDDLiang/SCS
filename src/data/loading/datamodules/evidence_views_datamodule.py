"""Preserve A or G/H stream contracts; change only history choice or easy allocation."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from lightning.pytorch.trainer.states import TrainerFn
from src.data.loading.datamodules.train_demand_allocation_datamodule import TrainDemandAllocationDataModule,file_sha256
from src.data.loading.datamodules.target_matched_views_datamodule import TargetMatchedViewsDataModule as GModule
from src.data.loading.datamodules.u5plus_views_datamodule import TargetMatchedViewsDataModule as HModule
from scripts.evidence_views_common import verify_protocol


def bind(module,name):
 r=module.evidence_manifest['files'][name];p=Path(r['path'])
 if file_sha256(p)!=r['sha256']:raise RuntimeError('Evidence asset SHA differs: '+str(p))
 a=np.load(p,mmap_mode='r',allow_pickle=False)
 if list(a.shape)!=r['shape'] or str(a.dtype)!=r['dtype']:raise RuntimeError('Evidence asset geometry differs')
 module.evidence_file_hashes[str(p)]=r['sha256'];return a


def initialize(module,path,mode):
 module.evidence_path=Path(path).resolve();module.evidence_mode=mode
 module.evidence_manifest_path=module.evidence_path/'manifest.json'
 module.evidence_identity=file_sha256(module.evidence_manifest_path)
 module.evidence_manifest=json.loads(module.evidence_manifest_path.read_text())
 m=module.evidence_manifest;module.evidence_protocol_sha=verify_protocol()
 if not m['ready'] or m['protocol_sha256']!=module.evidence_protocol_sha:raise RuntimeError('Evidence protocol/asset not ready')
 module.evidence_file_hashes={str(module.evidence_manifest_path):module.evidence_identity}
 module.evidence_counts=dict(input_changes_vs_reference=0,input_change_weight=0.,parent_changes_vs_reference=0,
  auxiliary_weight=0.,evidence_score_increment=0.,evidence_score_increment_weight=0.)
 module.pool_sha256=hashlib.sha256(json.dumps(dict(original=module.pool_sha256,evidence=module.evidence_identity,
  protocol=module.evidence_protocol_sha,mode=mode),sort_keys=True).encode()).hexdigest()


def extra_state(module):
 return dict(evidence_manifest_sha256=module.evidence_identity,evidence_protocol_sha256=module.evidence_protocol_sha,
  evidence_mode=module.evidence_mode,evidence_file_hashes=module.evidence_file_hashes,
  evidence_counts=copy.deepcopy(module.evidence_counts))


def validate_state(module,state):
 for key,value in extra_state(module).items():
  if key!='evidence_counts' and state.get(key)!=value:raise RuntimeError('Evidence checkpoint binding differs: '+key)


class InternalEvidenceViewsDataModule(TrainDemandAllocationDataModule):
 def __init__(self,*args,evidence_assets=None,evidence_mode='evidence',**kwargs):
  super().__init__(*args,**kwargs)
  if self.demand_mode!='random' or evidence_mode not in ('reference','evidence'):raise ValueError('Internal evidence requires exact old A random parents')
  initialize(self,evidence_assets,evidence_mode)
  if self.evidence_manifest['source_manifest_sha256']!=self.demand_identity:raise RuntimeError('Clean source binding differs')
  self.evidence_selected_views=bind(self,'selected_view_ids.npy');self.evidence_scores=bind(self,'view_scores.npy')
  if len(self.evidence_selected_views)!=len(self.view_key_ids):raise RuntimeError('Evidence selection map has wrong size')

 def _activate(self):
  first=not self.intervention_active
  super()._activate()
  if first:
   from scripts.evidence_prefix_guard import enforce
   enforce(self,True)

 def batch_details_at(self,index):
  d=super().batch_details_at(index);original=d['view_ids'].copy()
  d['reference_view_ids']=original
  if self.evidence_mode=='evidence':d['view_ids']=np.asarray(self.evidence_selected_views[original],np.int64)
  if not np.array_equal(self.view_key_ids[original],self.view_key_ids[d['view_ids']]):raise RuntimeError('Evidence altered internal parents')
  return d

 def _account(self,d,output):
  super()._account(d,output)
  ref=d['reference_view_ids'];selected=d['view_ids'];n=len(d['rows'])
  changes=int(np.count_nonzero(selected!=ref))
  gain=float(np.sum(self.evidence_scores[selected]-self.evidence_scores[ref]))
  e=self.evidence_counts;e['input_changes_vs_reference']+=changes;e['input_change_weight']+=changes/n
  e['auxiliary_weight']+=len(ref)/n;e['evidence_score_increment']+=gain;e['evidence_score_increment_weight']+=gain/n

 def state_dict(self):
  state=super().state_dict();state.update(extra_state(self));return state

 def load_state_dict(self,state):
  if getattr(getattr(self.trainer,'state',None),'fn',None) in (TrainerFn.TESTING,TrainerFn.VALIDATING,TrainerFn.PREDICTING):return
  validate_state(self,state);super().load_state_dict(state);self.evidence_counts=copy.deepcopy(state['evidence_counts'])


class FullEvidenceMixin:
 def __init__(self,*args,evidence_assets=None,evidence_mode='evidence',evidence_runtime=None,**kwargs):
  super().__init__(*args,**kwargs)
  if evidence_mode not in ('reference','evidence','easy'):raise ValueError('Unknown full evidence mode')
  initialize(self,evidence_assets,evidence_mode)
  m=self.evidence_manifest
  if m['original_views_manifest_sha256']!=self.target_view_identity or m['kind']!=self.relation_arm:raise RuntimeError('Original G/H binding differs')
  if m['statistics']['auxiliary_slots']!=len(self.copy_ids):raise RuntimeError('Original G/H slot count differs')
  self.reference_view_positions=self.view_positions;self.reference_view_changed=self.view_changed;self.reference_view_novel=self.view_novel
  self.evidence_selected_candidate=bind(self,'selected_candidate.npy')
  self.evidence_original_scores=bind(self,'original_score.npy');self.evidence_selected_scores=bind(self,'selected_score.npy')
  self.copy_parent_keys=bind(self,'copy_parent_keys.npy');self.copy_lengths=bind(self,'copy_lengths.npy')
  self.easy_runtime_path=Path(evidence_runtime).resolve() if evidence_runtime else None
  self.easy_runtime_identity=None;self.easy_ordinals=None
  if evidence_mode=='evidence':
   self.view_positions=bind(self,'selected_positions.npy');self.view_novel=bind(self,'selected_novel.npy')
  if evidence_mode=='easy':
   if self.relation_arm!='u2to4_x2' or self.easy_runtime_path is None:raise ValueError('Difficulty is limited to original G1/G2')
   if (self.easy_runtime_path/'manifest.json').exists():self._load_easy_runtime()

 def _load_easy_runtime(self):
  from scripts.prepare_easy_view_runtime import load_runtime
  load_runtime(self)

 def _activate(self):
  first=not self.intervention_active
  super()._activate()
  if first:
   from scripts.evidence_prefix_guard import enforce
   enforce(self,False)
  if first and self.evidence_mode=='easy':
   from scripts.prepare_easy_view_runtime import prepare_runtime
   prepare_runtime(self)

 def batch_details_at(self,index):
  d=super().batch_details_at(index)
  d['reference_ordinal']=d['ordinal'].copy();d['reference_ids']=d['ids'].copy()
  if self.evidence_mode=='easy' and index>=self.boundary:
   if self.easy_ordinals is None:raise RuntimeError('Easy allocation requires committed own-step200 TRAIN scores')
   oo=np.asarray(self.easy_ordinals[d['ordinal']],np.int64)
   d['ordinal']=oo;d['ids']=d['ids'].copy();d['ids'][d['auxiliary']]=self.copy_ids[oo]
   d['rows']=np.asarray(self.occurrences[d['ids']],np.int32).copy()
   d['keys']=np.asarray(self.occurrence_to_key[d['ids']],np.int64)
   d['changed'][d['auxiliary']]=self.reference_view_changed[oo]
   d['novel'][d['auxiliary']]=self.reference_view_novel[oo]
   if not np.array_equal(self.copy_lengths[oo],self.copy_lengths[d['reference_ordinal']]):raise RuntimeError('Easy allocation changed length slots')
  return d

 def on_before_batch_transfer(self,batch,dataloader_idx):
  training=self.trainer.training
  def indices(node):
   if isinstance(node,dict) and set(node)=={'relation_batch_index'}:return [int(node['relation_batch_index'])]
   if isinstance(node,(list,tuple)):return [i for child in node for i in indices(child)]
   return []
  ii=indices(batch) if training else []
  output=super().on_before_batch_transfer(batch,dataloader_idx)
  for index in ii:
   d=self.batch_details_at(index);n=len(d['rows']);oo=d['ordinal'];e=self.evidence_counts
   e['auxiliary_weight']+=len(oo)/n
   if self.evidence_mode=='evidence':
    changes=int(np.count_nonzero(self.evidence_selected_candidate[oo]))
    gain=float(np.sum(self.evidence_selected_scores[oo]-self.evidence_original_scores[oo]))
    e['evidence_score_increment']+=gain;e['evidence_score_increment_weight']+=gain/n
   elif self.evidence_mode=='easy' and len(oo):
    ref=super().batch_details_at(index)
    a=self.collate_details(d)[0].transformed_sequences['sequence_data']
    b=self.collate_details(ref)[0].transformed_sequences['sequence_data']
    changes=int(torch.any(a!=b,dim=1).sum())
    e['parent_changes_vs_reference']+=int(np.count_nonzero(d['ids']!=d['reference_ids']))
   else:changes=0
   e['input_changes_vs_reference']+=changes;e['input_change_weight']+=changes/n
  return output

 def state_dict(self):
  state=super().state_dict();state.update(extra_state(self))
  state['easy_runtime_manifest_sha256']=self.easy_runtime_identity
  return state

 def load_state_dict(self,state):
  if getattr(getattr(self.trainer,'state',None),'fn',None) in (TrainerFn.TESTING,TrainerFn.VALIDATING,TrainerFn.PREDICTING):return
  validate_state(self,state)
  if state.get('easy_runtime_manifest_sha256')!=self.easy_runtime_identity:raise RuntimeError('Easy runtime checkpoint binding differs')
  super().load_state_dict(state);self.evidence_counts=copy.deepcopy(state['evidence_counts'])


class FullEvidenceGDataModule(FullEvidenceMixin,GModule):pass
class FullEvidenceHDataModule(FullEvidenceMixin,HModule):pass
