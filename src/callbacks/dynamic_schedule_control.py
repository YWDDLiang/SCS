"""Independent update-boundary selector and incremental GPU occupancy budget."""
import time
import os
import hashlib
import json
import numpy as np
from pathlib import Path
import torch
import torch.distributed as dist
from lightning.pytorch import Callback
from src.callbacks.ddp_schedule_tracking import capture_rng, restore_rng, DDPScheduleCheckpoint
from src.callbacks.data_schedule_tracking import write_json


class DynamicScheduleControl(Callback):
    def __init__(self, arm, tracking, scorer=None, budget_gpu_hours=9.,
                 reserve_gpu_seconds=480., refresh_updates=200):
        self.arm=arm;self.tracking=tracking;self.scorer=scorer
        self.limit=budget_gpu_hours*3600.;self.reserve=reserve_gpu_seconds
        self.refresh_updates=refresh_updates;self.selector_state=None
        self.last_refresh=None;self.version=0;self.budget_stopped=False
        self.previous_increment=0.;self.prefix_counts=None;self.started=time.perf_counter()
        self.restored=False;self.score_records=[]
        self.bucket_audit=None
        self.hard_deadline=float(os.environ['DYNAMIC_HARD_DEADLINE_UNIX']) if os.environ.get('DYNAMIC_HARD_DEADLINE_UNIX') else None
        self.wrapper_started=float(os.environ['DYNAMIC_WRAPPER_STARTED_UNIX']) if os.environ.get('DYNAMIC_WRAPPER_STARTED_UNIX') else None
        self.segment_start_step=None;self.validation_suppressed=False
        self.resume_extra_gpu_seconds=float(os.environ.get('DYNAMIC_RESUME_EXTRA_GPU_SECONDS','0'))
        if not np.isfinite(self.resume_extra_gpu_seconds) or self.resume_extra_gpu_seconds<0:
            raise ValueError('Resume extra GPU seconds must be finite and nonnegative')

    def state_dict(self):
        return dict(selector_state=self.selector_state,last_refresh=self.last_refresh,
                    version=self.version,previous_increment=getattr(self,'checkpoint_increment',self.elapsed()),
                    prefix_counts=self.prefix_counts,score_records=self.score_records,bucket_audit=self.bucket_audit,
                    resume_extra_gpu_seconds_applied=self.resume_extra_gpu_seconds)

    def load_state_dict(self,state):
        self.selector_state=state['selector_state'];self.last_refresh=state['last_refresh']
        self.version=state['version'];self.previous_increment=state['previous_increment']+self.resume_extra_gpu_seconds
        self.prefix_counts=state['prefix_counts'];self.score_records=state['score_records']
        self.restored=True

    def elapsed(self):
        return self.previous_increment+(4*(time.perf_counter()-self.started) if self.started else 0.)

    def used(self,trainer):
        # Sum actual rank process wall; MAX is only used for consistent safety decisions.
        current=time.time()-self.wrapper_started if self.wrapper_started is not None else time.perf_counter()-self.started
        local=self.previous_increment/4+current
        tensor=torch.tensor(local,dtype=torch.float64,device=trainer.strategy.root_device)
        dist.all_reduce(tensor,op=dist.ReduceOp.SUM)
        return tensor.item()

    def remaining(self,trainer,used):
        available=self.limit-used
        if self.hard_deadline is not None:available=min(available,4*(self.hard_deadline-time.time()))
        tensor=torch.tensor(available,dtype=torch.float64,device=trainer.strategy.root_device)
        dist.all_reduce(tensor,op=dist.ReduceOp.MIN)
        return tensor.item()

    def epsilon(self,step):
        return max(.5,.75-(max(200,step)-200)*.25/400)

    def refresh(self,trainer,model):
        if self.arm=='R':return
        if self.scorer is None:raise RuntimeError('F/D require an explicit scorer')
        rng=capture_rng();torch.cuda.synchronize();start=time.perf_counter()
        try:
            state=self.scorer.score(model=model,optimizer=trainer.optimizers[0],datamodule=trainer.datamodule,
                global_step=trainer.global_step,previous_state=self.selector_state)
            if not isinstance(state,dict):raise TypeError('scorer must return serializable state dict')
            self.selector_state=state;self.version+=1;self.last_refresh=trainer.global_step
            trainer.datamodule.set_selection_weights(state['weights'],version=self.version,
                epsilon=self.epsilon(trainer.global_step))
        finally:
            torch.cuda.synchronize();restore_rng(rng)
        if trainer.is_global_zero:
            def summary(values):
                x=values.detach().cpu().numpy()
                return dict(count=len(x),mean=float(x.mean(dtype=np.float64)),
                    quantile_levels=[0,.1,.25,.5,.75,.9,1],
                    quantiles=np.quantile(x,[0,.1,.25,.5,.75,.9,1]).tolist(),
                    positive_fraction=float((x>0).mean()))
            journal=dict(step=trainer.global_step,version=self.version,epsilon=self.epsilon(trainer.global_step),
                metadata=state['metadata'],statistics={key:summary(state[key]) for key in ['losses','progress','alignment_raw']},
                policy='Summary only; full vectors live solely in rolling best/final checkpoints')
            folder=self.tracking.output_dir/'score_refresh';folder.mkdir(exist_ok=True)
            write_json(folder/f'step_{trainer.global_step:08d}_version{self.version}.json',journal)
        trainer.strategy.barrier('dynamic_score_journal_committed')
        seconds=time.perf_counter()-start
        self.tracking.local['score_device_seconds']+=seconds
        self.tracking.last_batch_end=time.perf_counter() # scoring is not loader wait
        self.score_records.append(dict(step=trainer.global_step,version=self.version,
                                       local_synchronized_wall_seconds=seconds))
        # Preserve F/D scorer provenance/loss history in the selector checkpoint.
        self.tracking.snapshot(trainer)

    def on_train_start(self,trainer,model):
        # Tracking callback runs first and restores per-rank RNG/data/Adam before here.
        self.segment_start_step=trainer.global_step
        if self.resume_extra_gpu_seconds and not self.restored:
            raise RuntimeError('Extra resume charge requires a saved dynamic budget state, not a fresh warmup branch')
        self.audit_bucket(trainer)
        if self.prefix_counts is None:self.prefix_counts=dict(self.tracking.global_counts)
        if not self.restored and trainer.global_step!=200:
            raise RuntimeError('Initial dynamic branch requires actual shared warmup200 final')
        if self.selector_state is not None:
            trainer.datamodule.set_selection_weights(self.selector_state['weights'],version=self.version,
                epsilon=self.epsilon(trainer.global_step))
        else:trainer.datamodule.set_selection_weights(None,version=self.version,epsilon=self.epsilon(trainer.global_step))
        if self.arm!='R' and self.last_refresh is None:self.refresh(trainer,model)
        self.last_used=self.used(trainer)
        if self.remaining(trainer,self.last_used)<=self.reserve:
            self.request_budget_stop(trainer)

    def request_budget_stop(self,trainer):
        """Prevent Lightning should_stop from selecting an off-cadence model."""
        interval=trainer.val_check_batch
        legal=(trainer.global_step>self.segment_start_step and np.isfinite(interval)
               and (trainer.global_step*trainer.accumulate_grad_batches)%interval==0)
        if not legal:
            # enable_validation consults this value; test has its own independent
            # limit_test_batches. Do not disable a new legal scheduled validation.
            trainer.limit_val_batches=0
            self.validation_suppressed=True
        self.budget_stopped=True;trainer.should_stop=True

    def audit_bucket(self,trainer):
        ddp=trainer.model
        self.bucket_audit=dict(find_unused_parameters=ddp.find_unused_parameters,static_graph=ddp.static_graph,
            has_rebuilt_buckets=ddp._has_rebuilt_buckets,bucket_sizes=ddp._get_ddp_logging_data().get('bucket_sizes'))
        if not self.bucket_audit['find_unused_parameters'] or self.bucket_audit['static_graph'] or self.bucket_audit['has_rebuilt_buckets']:
            raise RuntimeError('Dynamic fixed initial-bucket reduction policy was not respected')

    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        if self.tracking.local['microbatches']%32:return
        self.audit_bucket(trainer)
        used=self.used(trainer)
        # Reserve one observed update and (when due) one observed scoring round,
        # in addition to test/save reserve. Stop decisions are collective boundaries.
        since_last=max(0.,used-getattr(self,'last_used',0.))
        self.last_used=used
        next_score=(4*self.score_records[-1]['local_synchronized_wall_seconds']
                    if self.score_records and self.arm=='D' and (trainer.global_step-200)%self.refresh_updates==0 else 0.)
        forecast=torch.tensor(next_score,dtype=torch.float64,device=trainer.strategy.root_device)
        dist.all_reduce(forecast,op=dist.ReduceOp.MAX)
        next_score=forecast.item()
        if self.remaining(trainer,used)<=self.reserve+since_last+next_score:
            self.request_budget_stop(trainer);return
        if self.selector_state is not None:
            trainer.datamodule.set_selection_weights(self.selector_state['weights'],version=self.version,
                epsilon=self.epsilon(trainer.global_step))
        if self.arm=='D' and trainer.global_step>=200+self.refresh_updates and (trainer.global_step-200)%self.refresh_updates==0 and self.last_refresh!=trainer.global_step:
            self.refresh(trainer,model)

    def on_train_batch_start(self,trainer,model,batch,batch_idx):
        actual=getattr(trainer.datamodule,'last_consumed_selection_version',None)
        if actual is None:raise RuntimeError('Dynamic DM must expose consumed selection version for prefetched-batch audit')
        if int(actual)!=self.version:
            raise RuntimeError(f'Stale selection version consumed: actual={actual}, expected={self.version}, step={trainer.global_step}')
        if self.tracking.kind=='pilot':
            ids=trainer.datamodule.last_consumed_candidate_indices
            values=torch.as_tensor(ids).detach().cpu().contiguous().numpy()
            record=dict(rank=trainer.global_rank,global_index=trainer.datamodule.last_consumed_global_index,
                        version=self.version,candidate_count=len(values),
                        candidate_indices_sha256=hashlib.sha256(values.tobytes()).hexdigest())
            with (self.tracking.output_dir/f'pilot_sampler_rank{trainer.global_rank}.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')

    def budget_summary(self,trainer):
        used=self.used(trainer)
        effective=min(self.limit,self.previous_increment+4*(self.hard_deadline-self.wrapper_started)) if self.hard_deadline is not None and self.wrapper_started is not None else self.limit
        return dict(limit_gpu_seconds=self.limit,nominal_gpu_hours=self.limit/3600,effective_clipped_gpu_seconds=effective,
                    hard_deadline_unix=self.hard_deadline,wrapper_started_unix=self.wrapper_started,
                    remaining_gpu_seconds=self.remaining(trainer,used),reserve_gpu_seconds=self.reserve,
                    measured_incremental_gpu_process_wall_seconds=used,
                    excludes_shared_warmup=True,budget_stopped=self.budget_stopped,
                    prefix_counts=self.prefix_counts,score_records=self.score_records,bucket_audit=self.bucket_audit,
                    segment_start_step=self.segment_start_step,validation_suppressed_at_budget_stop=self.validation_suppressed,
                    resume_extra_gpu_seconds_applied=self.resume_extra_gpu_seconds)


class DynamicScheduleCheckpoint(DDPScheduleCheckpoint):
    """Same inherited roles/state identity, with all-rank budget snapshot at save."""
    def __init__(self,output_dir,tracking,control):
        super().__init__(output_dir,tracking)
        self.control=control
        self.legacy_state_key=DDPScheduleCheckpoint(output_dir,tracking).state_key

    @property
    def state_key(self):
        return self.legacy_state_key

    def save_final(self,trainer,is_best=False):
        self.control.checkpoint_increment=self.control.used(trainer)
        return super().save_final(trainer,is_best)

    def on_fit_end(self,trainer,pl_module):
        branch_final=(self.final_checkpoint and Path(self.final_checkpoint).parent.resolve()==Path(self.dirpath).resolve())
        if self.final_step!=trainer.global_step or not branch_final:
            # Initial step200 or same-step continuation can change selector/data
            # state without an optimizer update. Never leave final pointing to the
            # shared warmup. If best is this model, rebind both roles to one new file.
            self.save_final(trainer,is_best=self.best_step==trainer.global_step)
