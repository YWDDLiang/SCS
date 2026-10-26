"""Native-quota dynamic data scheduling; utility is applied only at consumption."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from lightning.pytorch.trainer.states import TrainerFn
from src.data.loading.datamodules.ddp_scheduled_datamodule import DDPScheduledSequenceDataModule
from src.data.loading.components.cached_eval_batches import _identity
from src.data.loading.components.fast_scheduled_collate import FastScheduledCollator
from src.data.selection.subsequence_candidate import SubsequenceCandidateStore


def _array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


class _ProposalBatches(IterableDataset):
    def __init__(self,module):self.module=module
    def __iter__(self):
        module=self.module;cursor=module.suffix_consumed_batches;rank=module.trainer.global_rank
        # Read only proposal metadata during Lightning lookahead. No sampler RNG,
        # utility, candidate draw or model input is touched before transfer.
        while True:
            yield {'dynamic_proposal':module.proposal_at(module.source_offset_batches+cursor+rank),
                   'suffix_global_batch_index':cursor+rank}
            cursor+=4


class DynamicScheduleDataModule(DDPScheduledSequenceDataModule):
    def __init__(self,*args,assets_path,sampler_seed=42,source_offset_batches=25600,scheduler_arm='R',epsilon=.75,**kwargs):
        super().__init__(*args,**kwargs)
        if scheduler_arm not in ('R','F','D'):raise ValueError('scheduler_arm must be R/F/D')
        if source_offset_batches%128:raise ValueError('Source origin must be an optimizer boundary')
        self.assets_path=Path(assets_path);self.manifest=json.loads((self.assets_path/'manifest.json').read_text())
        self.assets=torch.load(self.manifest['groups_path'],map_location='cpu',weights_only=True)
        if hashlib.sha256(Path(self.manifest['groups_path']).read_bytes()).hexdigest()!=self.manifest['groups_sha256']:
            raise RuntimeError('Dynamic asset identity mismatch')
        self.source_trace_dir=self.manifest['source_trace_dir'];self.candidate_store_path=self.manifest['candidate_store_path'];self.native_source_dir=None
        self.store=SubsequenceCandidateStore.load(self.candidate_store_path)
        self.scheduler_arm=scheduler_arm;self.sampler_seed=int(sampler_seed);self.source_offset_batches=int(source_offset_batches)
        self.suffix_consumed_batches=0;self.consumed_batches=self.source_offset_batches
        self.group_ids=self.assets['candidate_group_ids'];self._groups=self.group_ids.numpy()
        self._order=self.assets['candidate_indices'].numpy();self._offsets=self.assets['group_offsets'].numpy()
        self._prior=self.assets['sampling_prior'].numpy();self._prior_mass=np.bincount(self._groups,weights=self._prior,minlength=len(self._offsets)-1)
        self._base_cdf=self._group_cdf(self._prior);self._utility_cdf=self._base_cdf
        self.selection_version=0;self.epsilon=float(epsilon);self._utility=np.ones(self.store.num_candidates,dtype=np.float64)
        self.selection_weights_hash=_array_sha(self._utility)
        self._chunk_index=None;self._chunk_payload=None;self._collator=None
        self._uid_sort=np.argsort(self.store.user_ids.numpy());self._sorted_uids=self.store.user_ids.numpy()[self._uid_sort]
        per_user=np.bincount(self.store.candidate_triples[:,0].numpy(),minlength=self.store.num_rows);self._user_offsets=np.r_[0,np.cumsum(per_user)]
        self.trace_batch_count=int(self.manifest['source_consumed_batches'])
        if self.trace_batch_count%128 or not 0<=self.epsilon<=1:raise ValueError('Invalid frozen trace or epsilon')
        if self._source_identity()!=self.manifest['source_stream_id']:raise RuntimeError('Dynamic source UUID mismatch')
        self.selected_candidate_counts=np.zeros(self.store.num_candidates,dtype=np.int64)
        self.actual_proposal_occurrences=0;self.actual_selected_occurrences=0;self.slot_quota_violations=0
        self.exposure_complete=True;self.exposure_unmeasured_prefix_suffix_batches=0;self.exposure_needs_rank_restore=False
        self.last_consumed_selection_version=None;self.last_consumed_global_index=None;self.last_consumed_source_batch_index=None;self.last_consumed_candidate_indices=None

    def _group_cdf(self,mass):
        ordered=np.asarray(mass,dtype=np.float64)[self._order];cumulative=np.cumsum(ordered)
        groups=self._groups[self._order];nonempty=np.flatnonzero(np.diff(self._offsets)>0)
        starts=self._offsets[nonempty];before=np.zeros(len(self._offsets)-1,dtype=np.float64)
        has_before=starts>0;before[nonempty[has_before]]=cumulative[starts[has_before]-1]
        totals=np.bincount(self._groups,weights=mass,minlength=len(before))
        cdf=(cumulative-before[groups])/totals[groups]
        cdf[self._offsets[nonempty+1]-1]=1.
        return cdf

    def set_selection_weights(self,weights,version,epsilon):
        if not 0<=float(epsilon)<=1:raise ValueError('epsilon must be in [0,1]')
        self.epsilon=float(epsilon)
        if version==self.selection_version:return  # Fast epsilon-only update.
        if weights is None:
            if self.scheduler_arm!='R':raise ValueError('New F/D version requires explicit utility weights')
            self.selection_version=version;return
        values=(weights.detach().cpu().numpy() if torch.is_tensor(weights) else np.asarray(weights))
        if values.shape!=(self.store.num_candidates,) or not np.isfinite(values).all() or np.any(values<=0):
            raise ValueError('Utility must be one finite positive weight per canonical candidate')
        if self.scheduler_arm=='R' and not np.all(values==1):raise ValueError('R control must retain pure native prior')
        self._utility=np.ascontiguousarray(values,dtype=np.float64).copy();self.selection_weights_hash=_array_sha(self._utility)
        self._utility_cdf=self._group_cdf(self._prior*self._utility);self.selection_version=version

    def _prepare_collator(self):
        if self._collator is None:
            config=self.stage_to_config[TrainerFn.FITTING]
            self._collator=FastScheduledCollator(self.store,labels=config.labels,sequence_length=config.sequence_length,
                masking_token=config.masking_token,padding_token=config.padding_token,oov_token=config.get('oov_token',None))
        return self._collator

    def proposal_at(self,global_source_index):
        index=int(global_source_index)%self.trace_batch_count;chunk,within=divmod(index,128)
        if chunk!=self._chunk_index:
            name=f'chunk_{chunk:07d}.pt'
            self._chunk_payload=torch.load(Path(self.source_trace_dir)/name,map_location='cpu',weights_only=True);self._chunk_index=chunk
            if len(self._chunk_payload['batch_offsets'])-1!=128:raise RuntimeError('Dynamic source chunk does not contain128original batches')
        offsets=self._chunk_payload['batch_offsets'];start,end=int(offsets[within]),int(offsets[within+1])
        return self._chunk_payload['occurrences'][start:end]

    def _proposal_candidates(self,occurrences):
        values=occurrences.numpy().astype(np.int64);uid,raw_start,target=values.T
        locations=np.searchsorted(self._sorted_uids,uid)
        if np.any(locations>=len(self._sorted_uids)) or not np.array_equal(self._sorted_uids[locations],uid):raise RuntimeError('Proposal UID absent from store')
        rows=self._uid_sort[locations];minimum=np.maximum(target+1-30,0);effective=np.maximum(raw_start,minimum)
        prefix=np.minimum(target-1,29);prefix=prefix*(prefix+1)//2+np.maximum(target-30,0)*29
        indices=self._user_offsets[rows]+prefix+effective-minimum
        return indices,self._groups[indices]

    def _draw(self,occurrences,suffix_global_index):
        original,groups=self._proposal_candidates(occurrences)
        # A keyed private PCG64 stream: identical uniform variates for every arm,
        # independent of rank, prefetch, model RNG and utility scoring cadence.
        seed=np.random.SeedSequence([self.sampler_seed,int(suffix_global_index)&0xffffffff,int(suffix_global_index)>>32])
        rng=np.random.Generator(np.random.PCG64(seed));uniforms=rng.random(len(groups))
        low=self._offsets[groups].copy();high=self._offsets[groups+1]-1
        epsilon=1. if self.scheduler_arm=='R' else self.epsilon
        while np.any(low<high):
            mid=(low+high)//2;cdf=epsilon*self._base_cdf[mid]+(1-epsilon)*self._utility_cdf[mid]
            right=cdf<uniforms;active=low<high
            low=np.where(active&right,mid+1,low);high=np.where(active&~right,mid,high)
        picked=self._order[low].copy()
        single=np.diff(self._offsets)[groups]<=1;picked[single]=original[single]
        if not np.array_equal(self._groups[picked],groups):raise RuntimeError('Sampled group quota mismatch')
        return torch.from_numpy(picked)

    def train_dataloader(self):
        self._require_world();self._prepare_collator()
        if self.suffix_consumed_batches%128:raise RuntimeError('Dynamic train resume requires committed128boundary')
        return (DataLoader(_ProposalBatches(self),batch_size=None,num_workers=0,collate_fn=_identity,
                generator=torch.Generator().manual_seed(self.stream_seed),pin_memory=False),)

    def on_before_batch_transfer(self,batch,dataloader_idx):
        if not self.trainer.training:return batch
        def consume(node):
            if isinstance(node,dict) and set(node)=={'dynamic_proposal','suffix_global_batch_index'}:
                index=int(node['suffix_global_batch_index']);rank=self.trainer.global_rank
                if index%4!=rank or index-rank!=self.suffix_consumed_batches:raise RuntimeError('Dynamic rank cursor mismatch')
                if self.exposure_needs_rank_restore:raise RuntimeError('Restore this rank exposure state before dynamic sampling')
                version=self.selection_version;chosen=self._draw(node['dynamic_proposal'],index)
                output=self._prepare_collator().collate_candidate_indices(chosen)
                np.add.at(self.selected_candidate_counts,chosen.numpy(),1)
                self.actual_proposal_occurrences+=len(node['dynamic_proposal']);self.actual_selected_occurrences+=len(chosen)
                self.suffix_consumed_batches=index-rank+4;self.consumed_batches=self.source_offset_batches+self.suffix_consumed_batches
                self.last_consumed_selection_version=version;self.last_consumed_global_index=self.source_offset_batches+index
                self.last_consumed_source_batch_index=(self.source_offset_batches+index)%self.trace_batch_count;self.last_consumed_candidate_indices=chosen
                return output
            if not isinstance(node,(tuple,list)):return node
            children=[consume(child) for child in node]
            return tuple(children) if isinstance(node,tuple) else children
        return consume(batch)

    def scoring_batches(self,batch_size=512):
        collator=self._prepare_collator()
        for start in range(0,self.store.num_candidates,batch_size):
            indices=torch.arange(start,min(start+batch_size,self.store.num_candidates),dtype=torch.int64)
            yield collator.collate_candidate_indices(indices),indices

    def reference_batches(self,batch_size=8):
        collator=self._prepare_collator();reference=self.assets['reference_candidate_indices']
        for start in range(0,len(reference),batch_size):
            indices=reference[start:start+batch_size]
            yield collator.collate_candidate_indices(indices),indices

    def state_dict(self):
        if self.suffix_consumed_batches%128:raise RuntimeError('Dynamic checkpoint only at committed optimizer boundary')
        return {'dynamic_format_version':1,'world_size':4,'global_accumulation_batches':128,'mode':self.mode,
            'scheduler_arm':self.scheduler_arm,'source_stream_id':self.manifest['source_stream_id'],'assets_sha256':self.manifest['groups_sha256'],
            'global_committed_cursor':self.consumed_batches,'consumed_batches':self.consumed_batches,'suffix_consumed_batches':self.suffix_consumed_batches,
            'source_offset_batches':self.source_offset_batches,'sampler_rng':{'algorithm':'PCG64-keyed-SeedSequence-v1','seed':self.sampler_seed,'counter':self.suffix_consumed_batches},
            'selection_version':self.selection_version,'selection_weights':torch.from_numpy(self._utility.copy()),'selection_weights_hash':self.selection_weights_hash,
            'epsilon':self.epsilon,'stream_seed':self.stream_seed,
            'exposure_format_version':1,'exposure_rank':int(self.trainer.global_rank),
            'selected_candidate_counts':torch.from_numpy(self.selected_candidate_counts.copy()),
            'actual_proposal_occurrences':self.actual_proposal_occurrences,'actual_selected_occurrences':self.actual_selected_occurrences,
            'slot_quota_violations':self.slot_quota_violations,'exposure_complete':self.exposure_complete,
            'exposure_unmeasured_prefix_suffix_batches':self.exposure_unmeasured_prefix_suffix_batches}

    def load_state_dict(self,state):
        if getattr(getattr(self.trainer,'state',None),'fn',None) in (TrainerFn.TESTING,TrainerFn.VALIDATING,TrainerFn.PREDICTING):return
        if 'dynamic_format_version' not in state:
            if not self.restart_stream_on_restore:raise RuntimeError('Warmup ordinary DDP state is not dynamic sampler state; explicit first-branch restart required')
            if state.get('world_size')!=4 or state.get('source_stream_id')!=self.manifest['source_stream_id'] or int(state.get('global_committed_cursor',-1))!=self.source_offset_batches:
                raise RuntimeError('Warmup topology/source/update200 identity mismatch')
            self.suffix_consumed_batches=0;self.consumed_batches=self.source_offset_batches
            self.selected_candidate_counts.fill(0);self.actual_proposal_occurrences=0;self.actual_selected_occurrences=0;self.slot_quota_violations=0
            self.exposure_complete=True;self.exposure_unmeasured_prefix_suffix_batches=0;self.exposure_needs_rank_restore=False
            return
        if state.get('dynamic_format_version')!=1 or state.get('world_size')!=4 or state.get('scheduler_arm')!=self.scheduler_arm:
            raise RuntimeError('Dynamic checkpoint protocol mismatch')
        if state['source_stream_id']!=self.manifest['source_stream_id'] or state['assets_sha256']!=self.manifest['groups_sha256']:
            raise RuntimeError('Dynamic checkpoint source/assets identity mismatch')
        if state['source_offset_batches']!=self.source_offset_batches or state['stream_seed']!=self.stream_seed or state['sampler_rng']['seed']!=self.sampler_seed:
            raise RuntimeError('Dynamic checkpoint stream RNG/origin mismatch')
        suffix=int(state['suffix_consumed_batches'])
        if suffix%128 or state['global_committed_cursor']!=self.source_offset_batches+suffix or state['sampler_rng']['counter']!=suffix:raise RuntimeError('Dynamic saved cursor mismatch')
        self.selection_version=None
        self.set_selection_weights(state['selection_weights'],state['selection_version'],state['epsilon'])
        if self.selection_weights_hash!=state['selection_weights_hash']:raise RuntimeError('Dynamic utility hash mismatch')
        self.suffix_consumed_batches=suffix;self.consumed_batches=self.source_offset_batches+suffix
        if 'selected_candidate_counts' not in state:
            # Engineering probes predating exposure instrumentation remain valid
            # sampling resumes, but their unmeasured prefix is never fabricated.
            self.selected_candidate_counts.fill(0);self.actual_proposal_occurrences=0;self.actual_selected_occurrences=0;self.slot_quota_violations=0
            self.exposure_complete=False;self.exposure_unmeasured_prefix_suffix_batches=suffix;self.exposure_needs_rank_restore=False
        elif int(state['exposure_rank'])!=int(self.trainer.global_rank):
            # Lightning loads rank0's default DM state on every rank. The tracking
            # callback subsequently restores each rank's gathered data state.
            self.exposure_needs_rank_restore=True
        else:
            counts=state['selected_candidate_counts'].cpu().numpy()
            if counts.shape!=self.selected_candidate_counts.shape or np.any(counts<0):raise RuntimeError('Invalid saved exposure counters')
            self.selected_candidate_counts[:]=counts
            self.actual_proposal_occurrences=int(state['actual_proposal_occurrences']);self.actual_selected_occurrences=int(state['actual_selected_occurrences'])
            self.slot_quota_violations=int(state['slot_quota_violations']);self.exposure_complete=bool(state['exposure_complete'])
            self.exposure_unmeasured_prefix_suffix_batches=int(state.get('exposure_unmeasured_prefix_suffix_batches',0));self.exposure_needs_rank_restore=False
            if int(counts.sum())!=self.actual_selected_occurrences or self.actual_selected_occurrences!=self.actual_proposal_occurrences or self.slot_quota_violations:
                raise RuntimeError('Saved exposure count/quota consistency mismatch')

    def restart_data_stream(self):
        # First branch origin is update200; later dynamic resumes must keep state.
        self.suffix_consumed_batches=0;self.consumed_batches=self.source_offset_batches
