"""At update200, filter one immutable occurrence permutation without restarting."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,IterableDataset
from lightning.pytorch.trainer.states import TrainerFn
from src.data.loading.datamodules.replay_shuffle_datamodule import ReplayShuffleSequenceDataModule
from src.data.loading.components.cached_eval_batches import _identity


class _LazySelectionBatches(IterableDataset):
    def __init__(self,module):self.module=module
    def __iter__(self):
        cursor=self.module.consumed_batches;rank=self.module.trainer.global_rank
        while True:
            for index in range(cursor+rank,cursor+128,4):
                # Lightning lookahead can read this handle before update200;
                # selection and model-input construction happen only at transfer.
                yield {'replay_selection_batch_index':index}
            cursor+=128


def _state_hash(value):
    digest=hashlib.sha256()
    def visit(node):
        if torch.is_tensor(node):
            tensor=node.detach().cpu().contiguous();digest.update(str((str(tensor.dtype),tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(node,np.ndarray):
            digest.update(str((node.dtype.str,node.shape)).encode());digest.update(node.tobytes())
        elif isinstance(node,dict):
            for key in sorted(node,key=lambda item:(str(type(item)),str(item))):digest.update(repr(key).encode());visit(node[key])
        elif isinstance(node,(tuple,list)):
            for item in node:visit(item)
        else:digest.update(repr(node).encode())
    visit(value);return digest.hexdigest()


class ReplaySelectionSequenceDataModule(ReplayShuffleSequenceDataModule):
    def __init__(self,*args,selection_arm=None,selection_ratio=.8,selection_output_dir=None,selection_dataset=None,**kwargs):
        if float(selection_ratio)!=.8:raise ValueError('This experiment retains exactly floor(0.8*N) occurrences')
        # The completed full module remains unchanged: use its fullasset loading
        # and reconstruction, then restore the new scientific retention setting.
        super().__init__(*args,selection_ratio=1.,**kwargs)
        self.selection_ratio=.8
        self.selection_arm=selection_arm or os.environ.get('REPLAY_SELECTION_ARM')
        if self.selection_arm not in ('random80','gradient80'):raise ValueError('REPLAY_SELECTION_ARM must be random80 or gradient80')
        self.selection_boundary_batches=25600
        self.selection_output_dir=Path(selection_output_dir) if selection_output_dir else None;self.dataset=selection_dataset or self.asset_manifest.get('identity',{}).get('dataset');self.selected_positions=None;self.selection_manifest=None;self.selection_identity=None
        self._selection_start_offset=None

    def full_scan_position_at(self,batch_cursor):
        cycles,within=divmod(int(batch_cursor),self.batch_period)
        return cycles*self.occurrence_count+int(self.k_offsets[within])

    def _selection_manifest_file(self,folder):
        for name in ('selection_manifest.json','selected_manifest.json','manifest.json'):
            path=Path(folder)/name
            if path.exists():return path
        raise RuntimeError('Selection helper did not publish a ready manifest')

    def _bind_selection(self,folder):
        from src.data.selection.replay_occurrence_selection import load_selected_positions
        positions,manifest=load_selected_positions(Path(folder),self.assets_path)
        if positions.dtype!=np.int64 or positions.ndim!=1 or len(positions)!=4*self.occurrence_count//5:
            raise RuntimeError('Selected positions do not retain exact floor80% of original occurrence identities')
        self.selected_positions=positions;self.selection_manifest=manifest
        self.selection_identity=hashlib.sha256(self._selection_manifest_file(folder).read_bytes()).hexdigest()
        self.selection_folder=str(Path(folder).resolve())
        prefix_scan=self.full_scan_position_at(self.selection_boundary_batches)
        cycle,within=divmod(prefix_scan,self.occurrence_count)
        self._selection_start_offset=cycle*len(positions)+int(np.searchsorted(positions,within,side='left'))

    def _audit_prefix(self):
        started=time.monotonic();model=self.trainer.lightning_module
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
        if torch.cuda.is_initialized():rng['cuda_local']=torch.cuda.get_rng_state(torch.cuda.current_device())
        record={'rank':int(self.trainer.global_rank),'global_step':int(self.trainer.global_step),
            'committed_global_batch_cursor':self.consumed_batches,'full_scan_position':self.full_scan_position_at(self.consumed_batches),
            'model_sha256':_state_hash(model.state_dict()),'Adam_sha256':_state_hash(self.trainer.optimizers[0].state_dict()),
            'training_RNG_sha256':_state_hash(rng),'prefix_hash_CPU_wall_seconds':time.monotonic()-started}
        if dist.is_available() and dist.is_initialized():
            records=[None]*4;dist.all_gather_object(records,record)
        else:records=[record]
        if int(self.trainer.global_rank)==0:
            self.selection_output_dir.mkdir(parents=True,exist_ok=True)
            target=self.selection_output_dir/'prefix200_audit.json';temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps({'global_step':200,'data_assets_identity':self.pool_sha256,'selection_not_started':True,'ranks':records,
                'full_prefix_model_checkpoint_not_created':True,'full_curve_comparison_must_use_observed_validation_records':True},indent=2));temporary.replace(target)

    def ensure_selection(self):
        if self.selected_positions is not None:return
        if int(self.trainer.global_step)!=200 or self.consumed_batches!=self.selection_boundary_batches:
            raise RuntimeError('First selection must occur at the continuously trained update200 boundary')
        if self.selection_output_dir is None:self.selection_output_dir=Path(self.trainer.default_root_dir)/'selection'
        self._audit_prefix()
        module_name='src.data.selection.replay_occurrence_selection' if self.selection_arm=='random80' else 'src.data.selection.replay_gradient_selection'
        helper=importlib.import_module(module_name)
        folder=helper.ensure_selection(self)
        folder=Path(folder).resolve()
        if folder!=self.selection_output_dir.resolve():
            if int(self.trainer.global_rank)==0:
                source_manifest=self._selection_manifest_file(folder)
                target=self.selection_output_dir/'selected_positions.npy'
                if target.exists() or target.is_symlink():
                    if target.resolve()!=(folder/'selected_positions.npy').resolve():raise RuntimeError('Run selection already points to another mask')
                else:os.symlink(str(folder/'selected_positions.npy'),target)
                manifest_target=self.selection_output_dir/source_manifest.name
                if manifest_target.exists() and manifest_target.read_bytes()!=source_manifest.read_bytes():raise RuntimeError('Run selection manifest changed')
                manifest_target.write_bytes(source_manifest.read_bytes())
                (self.selection_output_dir/'selection_reference.json').write_text(json.dumps({'ready':True,'prepared_folder':str(folder),'manifest_sha256':hashlib.sha256(source_manifest.read_bytes()).hexdigest(),'source_stream_id':self.asset_source_stream_id,'selected_positions_sha256':json.loads(source_manifest.read_text())['files']['selected_positions.npy']['sha256']},indent=2))
            self.trainer.strategy.barrier('selection_run_manifest_published')
            folder=self.selection_output_dir
        self._bind_selection(folder)

    def selected_offset_at(self,batch_cursor):
        if self.selected_positions is None:raise RuntimeError('Selection not published')
        return self._selection_start_offset+self.full_scan_position_at(batch_cursor)-self.full_scan_position_at(self.selection_boundary_batches)

    def scan_position_at(self,batch_cursor):
        cursor=int(batch_cursor)
        if cursor<=self.selection_boundary_batches:return self.full_scan_position_at(cursor)
        offset=self.selected_offset_at(cursor);cycles,within=divmod(offset-1,len(self.selected_positions))
        return cycles*self.occurrence_count+int(self.selected_positions[within])+1

    def occurrence_indices_at(self,global_batch_index):
        index=int(global_batch_index)
        if index<self.selection_boundary_batches:return super().occurrence_indices_at(index)
        self.ensure_selection()
        offset=self.selected_offset_at(index);K=int(self.k_sizes[index%self.batch_period])
        positions=self.selected_positions[(offset+np.arange(K,dtype=np.int64))%len(self.selected_positions)]
        return np.asarray(self.permutation[positions],dtype=np.int64)

    def train_dataloader(self):
        # Reuse original store/collator preparation; do not instantiate another
        # training iterator or change the original model/DDP execution geometry.
        super().train_dataloader()
        config=self.stage_to_config[TrainerFn.FITTING]
        return (DataLoader(_LazySelectionBatches(self),batch_size=None,num_workers=0,collate_fn=_identity,
            pin_memory=False,generator=torch.Generator().manual_seed(self.stream_seed)),)

    def on_before_batch_transfer(self,batch,dataloader_idx):
        if not self.trainer.training:return batch
        def consume(node):
            if isinstance(node,dict) and set(node)=={'replay_selection_batch_index'}:
                index=int(node['replay_selection_batch_index']);rank=int(self.trainer.global_rank)
                if index%4!=rank or index-rank!=self.consumed_batches:raise RuntimeError('Selection replay rank cursor mismatch')
                records=self.occurrences_at(index)
                output=self._fast_collate(records)
                # Preserve the parent's exact nested tuple/cursor transfer rule.
                return super(ReplaySelectionSequenceDataModule,self).on_before_batch_transfer((*output,{'ddp_global_batch_index':index}),dataloader_idx)
            if not isinstance(node,(tuple,list)):return node
            values=[consume(value) for value in node]
            return tuple(values) if isinstance(node,tuple) else values
        return consume(batch)

    def state_dict(self):
        state=super().state_dict();active=self.selected_positions is not None
        state.update(replay_selection_format_version=1,selection_arm=self.selection_arm,selection_boundary_batches=25600,
            selection_ready=active,selection_folder=getattr(self,'selection_folder',None),selection_identity=self.selection_identity,
            original_P_scan_cursor=self.scan_position_at(self.consumed_batches) if active else self.full_scan_position_at(self.consumed_batches),
            selected_absolute_offset=self.selected_offset_at(self.consumed_batches) if active else None,
            permutation_position=(self.scan_position_at(self.consumed_batches) if active else self.full_scan_position_at(self.consumed_batches))%self.occurrence_count)
        return state

    def load_state_dict(self,state):
        if getattr(getattr(self.trainer,'state',None),'fn',None) in (TrainerFn.TESTING,TrainerFn.VALIDATING,TrainerFn.PREDICTING):return
        if state.get('replay_selection_format_version')!=1 or state.get('selection_arm')!=self.selection_arm or state.get('selection_boundary_batches')!=25600:
            raise RuntimeError('Full-shuffle or another arm checkpoint is not selection replay state')
        super().load_state_dict(state)
        if state['selection_ready']:
            self._bind_selection(state['selection_folder'])
            if state['selection_identity']!=self.selection_identity:raise RuntimeError('Selection mask manifest changed')
            if state['selected_absolute_offset']!=self.selected_offset_at(self.consumed_batches) or state['original_P_scan_cursor']!=self.scan_position_at(self.consumed_batches):
                raise RuntimeError('Selection scan/offset checkpoint mismatch')
        elif self.consumed_batches>25600:raise RuntimeError('Post200 checkpoint has no ready selection')
