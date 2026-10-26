"""Four-rank accounting and exactly two rolling checkpoint roles."""
import json
import os
import random
import resource
import time
from pathlib import Path
import numpy as np
import psutil
import torch
import torch.distributed as dist
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from src.callbacks.data_schedule_tracking import write_json, metrics_json


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(torch.cuda.current_device()) if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'],torch.cuda.current_device())


class RankMemoryReservation:
    """25GiB reusable cache on this rank's selected device; never on cuda:0 by assumption."""
    def __init__(self, local_rank):
        self.device=int(local_rank)
        self.minimum=int(os.environ.get('DATA_SELECT_MIN_GPU_MEMORY_MIB','25600'))*1024*1024
        self.refills=0
        self.observed={}
    def ensure(self):
        reserved=torch.cuda.memory_reserved(self.device)
        if reserved<self.minimum:
            block=torch.empty(self.minimum-reserved,dtype=torch.uint8,device=f'cuda:{self.device}')
            del block;self.refills+=1
        self.observed=dict(device=self.device,reserved_bytes=torch.cuda.memory_reserved(self.device),
            allocated_bytes=torch.cuda.memory_allocated(self.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),refills=self.refills)
        if self.observed['reserved_bytes']<self.minimum:raise RuntimeError('Rank reusable GPU memory floor not reached')
    def metadata(self):return dict(minimum_bytes=self.minimum,policy='rank-local reusable allocator cache',observed=self.observed)


class DDPScheduleTracking(Callback):
    def __init__(self, output_dir, kind='formal', memory_reservation=None):
        self.output_dir=Path(output_dir);self.kind=kind;self.memory=memory_reservation
        self.started=time.perf_counter();self.pending_rng=None;self.pending_rank_states=None
        self.local=dict(microbatches=0,examples=0,nonpadding_tokens=0,train_device_seconds=0.,
            validation_device_seconds=0.,test_device_seconds=0.,data_wait_seconds=0.,checkpoint_wall_seconds=0.,
            main_process_cpu_seconds=0.,worker_cpu_seconds=0.,gpu_process_wall_seconds=0.)
        self.global_counts=dict(self.local);self.cpu_prefix=0.;self.worker_prefix=0.;self.wall_prefix=0.
        self.last_batch_end=None;self.rank_facts=[];self.prepared_states=None

    def sync(self):
        if self.memory:self.memory.ensure()
        torch.cuda.synchronize(torch.cuda.current_device())

    def cpu_snapshot(self):
        self.local['main_process_cpu_seconds']=self.cpu_prefix+time.process_time()
        usage=resource.getrusage(resource.RUSAGE_CHILDREN);seconds=usage.ru_utime+usage.ru_stime
        for child in psutil.Process().children(recursive=True):
            try:
                t=child.cpu_times();seconds+=t.user+t.system
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        self.local['worker_cpu_seconds']=self.worker_prefix+seconds
        self.local['gpu_process_wall_seconds']=self.wall_prefix+time.perf_counter()-self.started

    def snapshot(self, trainer):
        self.cpu_snapshot()
        keys=list(self.local)
        values=torch.tensor([self.local[k] for k in keys],dtype=torch.float64,device=trainer.strategy.root_device)
        dist.all_reduce(values,op=dist.ReduceOp.SUM)
        self.global_counts={k:float(v) for k,v in zip(keys,values.cpu().tolist())}
        for k in ['microbatches','examples','nonpadding_tokens']:self.global_counts[k]=int(self.global_counts[k])
        return self.global_counts

    def state_dict(self):return {'global_counts':self.global_counts}
    def load_state_dict(self,state):self.global_counts=dict(state['global_counts'])

    def prepare_checkpoint(self,trainer):
        self.snapshot(trainer)
        payload=dict(rank=trainer.global_rank,rng=capture_rng(),data_state=trainer.datamodule.state_dict(),local_counts=dict(self.local))
        self.prepared_states=[None]*trainer.world_size
        dist.all_gather_object(self.prepared_states,payload)

    def on_save_checkpoint(self,trainer,pl_module,checkpoint):
        if self.prepared_states is None:raise RuntimeError('All ranks must prepare checkpoint RNG/data state before saving')
        checkpoint['ddp_schedule_rank_states']=self.prepared_states
        checkpoint['ddp_schedule_world_size']=trainer.world_size
        checkpoint['ddp_schedule_cost_definition']='GPU phase wall seconds summed across ranks; process wall additionally includes source wait and checkpoint/controller idle; CPU source cost separate'

    def on_load_checkpoint(self,trainer,pl_module,checkpoint):
        states=checkpoint.get('ddp_schedule_rank_states')
        if checkpoint.get('ddp_schedule_world_size')!=4 or not states or len(states)!=4:
            raise RuntimeError('Exact four-rank continuation requires all-rank RNG/data states; single-GPU checkpoints are not DDP continuations')
        self.pending_rank_states=states

    def on_train_start(self,trainer,pl_module):
        if trainer.world_size!=4:raise RuntimeError('DDP schedule requires exactly four ranks')
        if self.pending_rank_states:
            state=self.pending_rank_states[trainer.global_rank]
            if state['rank']!=trainer.global_rank:raise RuntimeError('Checkpoint rank-state order mismatch')
            self.local.update(state['local_counts']);trainer.datamodule.load_state_dict(state['data_state'])
            self.pending_rng=state['rng']
        self.cpu_prefix=self.local['main_process_cpu_seconds'];self.worker_prefix=self.local['worker_cpu_seconds'];self.wall_prefix=self.local['gpu_process_wall_seconds']
        if self.pending_rng:restore_rng(self.pending_rng);self.pending_rng=None
        self.sync();self.last_batch_end=time.perf_counter()
        properties=torch.cuda.get_device_properties(torch.cuda.current_device())
        fact=dict(rank=trainer.global_rank,local_rank=int(os.environ['LOCAL_RANK']),pid=os.getpid(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            gpu_name=properties.name,gpu_uuid=str(getattr(properties,'uuid','unknown')),memory=self.memory.metadata() if self.memory else None)
        self.rank_facts=[None]*4;dist.all_gather_object(self.rank_facts,fact)
        self.snapshot(trainer);self.status(trainer,'running')

    def status(self,trainer,status,**extra):
        if trainer.is_global_zero:
            write_json(self.output_dir/'status.json',dict(status=status,kind=self.kind,pid=os.getpid(),world_size=4,
                global_step=trainer.global_step,counts=self.global_counts,ranks=self.rank_facts,**extra))

    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        self.sync();now=time.perf_counter()
        if self.last_batch_end is not None:self.local['data_wait_seconds']+=now-self.last_batch_end
        self.batch_start=now
        data=batch
        while isinstance(data,(list,tuple)) and data:data=data[0]
        if hasattr(data,'transformed_sequences'):data=data.transformed_sequences
        if isinstance(data,dict):
            tensor=data.get('sequence_data',data.get('input_ids'))
            if torch.is_tensor(tensor):
                self.local['examples']+=tensor.shape[0];self.local['nonpadding_tokens']+=int((tensor!=-1).sum().item())

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        self.sync();self.last_batch_end=time.perf_counter()
        self.local['train_device_seconds']+=self.last_batch_end-self.batch_start;self.local['microbatches']+=1
        if self.local['microbatches']%32==0:
            self.snapshot(trainer);self.status(trainer,'running')

    def on_validation_start(self,trainer,pl_module):self.sync();self.eval_start=time.perf_counter()
    def on_test_start(self,trainer,pl_module):self.sync();self.eval_start=time.perf_counter()

    def per_user(self,trainer,pl_module,split):
        # Every rank calls this collective exactly once after its finite shard.
        user_ids,values=pl_module.evaluator.gathered_per_user_ndcg10()
        if trainer.is_global_zero:
            folder=self.output_dir/'per_user'/split;folder.mkdir(parents=True,exist_ok=True)
            step=getattr(self,'test_observation_step',trainer.global_step) if split=='test' else trainer.global_step
            torch.save(dict(global_step=step,split=split,metric='ndcg@10',user_ids=user_ids,values=values,
                mean=float(values.double().mean())),folder/f'step_{step:08d}.pt')

    def on_validation_end(self,trainer,pl_module):
        self.sync();self.local['validation_device_seconds']+=time.perf_counter()-self.eval_start
        self.snapshot(trainer)
        # callback_metrics can itself compute distributed metrics: read on all ranks.
        self.pending_validation=dict(step=trainer.global_step,quality=metrics_json(trainer.callback_metrics))
        self.per_user(trainer,pl_module,'validation');self.last_batch_end=time.perf_counter()

    def on_test_end(self,trainer,pl_module):
        self.sync();self.local['test_device_seconds']+=time.perf_counter()-self.eval_start
        self.per_user(trainer,pl_module,'test');self.snapshot(trainer)


class DDPScheduleCheckpoint(ModelCheckpoint):
    """Only best.ckpt and rolling final.ckpt; shared step uses best.ckpt alone."""
    def __init__(self,output_dir,tracking):
        super().__init__(dirpath=str(Path(output_dir)/'checkpoints'),monitor='val/ndcg@10',mode='max',save_top_k=0,save_last=False)
        self.tracking=tracking;self.best_step=None;self.final_step=None;self.final_checkpoint='';self.best_model_path='';self.best_model_score=None

    def state_dict(self):
        return dict(best_step=self.best_step,final_step=self.final_step,final_checkpoint=self.final_checkpoint,
            best_model_path=self.best_model_path,best_model_score=self.best_model_score)
    def load_state_dict(self,state):
        self.best_step=state['best_step'];self.final_step=state['final_step'];self.final_checkpoint=state['final_checkpoint']
        self.best_model_path=state['best_model_path'];self.best_model_score=state['best_model_score']

    def save_final(self,trainer,is_best=False):
        folder=Path(self.dirpath)
        if trainer.is_global_zero:folder.mkdir(parents=True,exist_ok=True)
        self.final_step=trainer.global_step
        if is_best:
            self.best_step=trainer.global_step;self.best_model_path=str(folder/'best.ckpt');self.final_checkpoint=self.best_model_path
        else:self.final_checkpoint=str(folder/'final.ckpt')
        trainer.strategy.barrier('ddp_checkpoint_role_ready')
        self.tracking.prepare_checkpoint(trainer)
        started=time.perf_counter()
        temporary=self.final_checkpoint+'.pending'
        trainer.save_checkpoint(temporary)
        if trainer.is_global_zero:
            os.replace(temporary,self.final_checkpoint)
            if is_best:(folder/'final.ckpt').unlink(missing_ok=True)
        trainer.strategy.barrier('ddp_checkpoint_atomic_commit')
        self.tracking.local['checkpoint_wall_seconds']+=time.perf_counter()-started
        self.tracking.snapshot(trainer)
        if trainer.is_global_zero:
            write_json(self.tracking.output_dir/'checkpoint_roles.json',dict(best=self.best_model_path or None,best_step=self.best_step,
                final=self.final_checkpoint,final_step=self.final_step,same_physical_checkpoint=self.best_step==self.final_step))

    def on_validation_end(self,trainer,pl_module):
        quality=metrics_json(trainer.callback_metrics);score=quality.get('val/ndcg@10')
        improved=score is not None and (self.best_model_score is None or score>float(self.best_model_score))
        if improved:self.best_model_score=torch.tensor(score,dtype=torch.float32)
        self.save_final(trainer,improved)
        if trainer.is_global_zero:
            record=dict(self.tracking.pending_validation,counts=self.tracking.global_counts,checkpoint=self.final_checkpoint,
                elapsed_seconds=time.perf_counter()-self.tracking.started,gpu_device_seconds=sum(self.tracking.global_counts[k] for k in ['train_device_seconds','validation_device_seconds','test_device_seconds']))
            with (self.tracking.output_dir/'validation_curve.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        self.tracking.last_batch_end=time.perf_counter();self.tracking.status(trainer,'running')

    def on_fit_end(self,trainer,pl_module):
        if self.final_step!=trainer.global_step:self.save_final(trainer,False)
    def on_train_epoch_end(self,trainer,pl_module):pass
    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):pass
