#!/usr/bin/env python
"""Four-GPU serial-task TIGER schedule runner. Launch with torchrun nproc_per_node=4."""
import os
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
import lightning as L
import hydra
from omegaconf import OmegaConf,open_dict
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from scripts.run_data_schedule_experiment import build_config as build_single_config
from src.callbacks.ddp_schedule_tracking import DDPScheduleTracking,DDPScheduleCheckpoint,RankMemoryReservation,capture_rng
from src.callbacks.data_schedule_tracking import write_json,metrics_json


def build_config(args):
    cfg=build_single_config(args)
    with open_dict(cfg):
        cfg.seed=42
        cfg.trainer.devices=4
        cfg.trainer.strategy={'_target_':'lightning.pytorch.strategies.DDPStrategy',
            'process_group_backend':'nccl','broadcast_buffers':False,'find_unused_parameters':False}
        cfg.trainer.accumulate_grad_batches=32
        cfg.trainer.val_check_interval=args.validation_updates*32
        cfg.trainer.check_val_every_n_epoch=None
        cfg.trainer.use_distributed_sampler=False
        cfg.trainer.deterministic=True
        cfg.data_loading.datamodule._target_='src.data.loading.datamodules.ddp_scheduled_datamodule.DDPScheduledSequenceDataModule'
        cfg.data_loading.datamodule.selection_weights_path=args.selection_weights
        cfg.data_loading.datamodule.global_accumulation_batches=128
    OmegaConf.resolve(cfg)
    return cfg


def parser_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['beauty','toys','sports'])
    p.add_argument('--mode',required=True,choices=['native_dynamic','native_replay','global_uniform'])
    p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,choices=[42],default=42)
    p.add_argument('--retention',type=float,default=1.)
    p.add_argument('--candidate-store');p.add_argument('--data-root');p.add_argument('--semantic-id-path')
    p.add_argument('--native-source-dir');p.add_argument('--source-trace-dir');p.add_argument('--follow-source-trace',action='store_true')
    p.add_argument('--eval-cache-dir',required=True);p.add_argument('--fast-data',action='store_true');p.add_argument('--efficient-eval',action='store_true')
    p.add_argument('--loader-context',default='spawn',choices=['spawn']);p.add_argument('--loader-timeout',type=int,default=600)
    p.add_argument('--selected-indices');p.add_argument('--checkpoint');p.add_argument('--restart-data-stream',action='store_true')
    p.add_argument('--selection-weights', help='Optional positive sampling weights for global_uniform candidate draws')
    p.add_argument('--pilot-updates',type=int);p.add_argument('--max-updates',type=int)
    p.add_argument('--validation-updates',type=int,default=50)
    p.add_argument('--config-only',action='store_true')
    args=p.parse_args();args.accumulation=128
    if not 0<args.retention<=1:p.error('retention must be in (0,1]')
    if args.mode=='native_dynamic' and not args.native_source_dir:p.error('DDP native requires persistent --native-source-dir')
    if args.mode!='native_dynamic' and not args.source_trace_dir:p.error('DDP replay/global requires --source-trace-dir')
    if args.mode=='native_dynamic' and args.fast_data:p.error('native consumes original batches; fast-data is replay/global only')
    if args.follow_source_trace and args.mode!='native_replay':p.error('follow is replay only')
    if args.validation_updates<=0:p.error('validation interval must be positive')
    return args


def main():
    args=parser_args();rank=int(os.environ.get('RANK','0'));local_rank=int(os.environ.get('LOCAL_RANK','0'))
    out=Path(args.output_dir).resolve()
    if rank==0:out.mkdir(parents=True,exist_ok=True)
    cfg=build_config(args);kind='pilot' if args.pilot_updates or args.max_updates else 'formal'
    metadata=dict(arguments=vars(args),kind=kind,world_size=4,seed=42,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),launched_unix=time.time(),
        global_accumulation_batches=128,local_accumulation_batches=32,global_validation_updates=args.validation_updates,
        local_validation_microbatches=args.validation_updates*32,precision='32-true',matmul_precision='medium',
        deterministic_algorithms=True,cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        checkpoint_policy='Only best.ckpt and rolling final.ckpt; same-step best/final uses best.ckpt alone; no eligible or step200 extra weights',
        source_cost_scope='Persistent CPU producer cost must be added separately, phase GPU seconds sum four ranks; model process wall includes wait/controller',
        evaluation_policy='Original evaluator global sum at epoch-end and per-user collective; cached batches sharded without padding; DDP broadcast_buffers=False',
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['scripts/run_data_schedule_ddp.py','src/callbacks/ddp_schedule_tracking.py','src/data/loading/datamodules/ddp_scheduled_datamodule.py']})
    if args.selection_weights:
        weight_path=Path(args.selection_weights)
        if not weight_path.exists():
            raise FileNotFoundError(weight_path)
        metadata['selection_weights_sha256']=hashlib.sha256(weight_path.read_bytes()).hexdigest()
    if rank==0:
        OmegaConf.save(cfg,out/'config.yaml');write_json(out/'launch_metadata.json',metadata)
    if args.config_only:
        if rank==0:write_json(out/'status.json',dict(status='configured',kind=kind,world_size=4))
        return
    if int(os.environ.get('WORLD_SIZE','0'))!=4 or not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=4:
        raise RuntimeError('Run under torchrun --nproc_per_node=4 with exactly four parent-allocated visible GPUs')
    # No CUDA allocation or RNG capture before selecting this rank's own device.
    torch.cuda.set_device(local_rank)
    memory=RankMemoryReservation(local_rank)
    tracking=DDPScheduleTracking(out,kind,memory)
    early=EarlyStopping(monitor='val/ndcg@10',mode='max',patience=10,min_delta=0.,check_on_train_epoch_end=False,check_finite=True)
    checkpoint=DDPScheduleCheckpoint(out,tracking)
    trainer=module=None
    try:
        memory.ensure();L.seed_everything(42,workers=True)
        model=hydra.utils.instantiate(cfg.model)
        tracking.pending_rng=capture_rng()
        module=hydra.utils.instantiate(cfg.data_loading.datamodule)
        trainer=hydra.utils.instantiate(cfg.trainer,callbacks=[early,tracking,checkpoint],
            logger=CSVLogger(str(out),name='csv'),enable_checkpointing=True)
        trainer.fit(model=model,datamodule=module,ckpt_path=args.checkpoint)
        memory.ensure()
        # Every rank reads metrics in identical order; computing them may be collective.
        quality=metrics_json(trainer.callback_metrics)
        reason='pilot' if kind=='pilot' else ('nonfinite' if quality.get('val/ndcg@10') is None else
            ('early_stopping' if trainer.should_stop and early.wait_count>=early.patience else 'max_steps'))
        stop_step=trainer.global_step;test=[]
        if kind=='formal' and reason=='early_stopping':
            if not checkpoint.best_model_path:raise RuntimeError('Early-stopped DDP run lacks best checkpoint')
            dmstate=module.state_dict()
            if rank==0:
                write_json(out/'fit_done.json',dict(stop_reason=reason,kind=kind,global_step=stop_step,
                    consumed_batches=module.consumed_batches,
                    native_source_stream_id=dmstate.get('source_stream_id',dmstate.get('native_source_stream_id',dmstate.get('source_trace_stream_id')))))
            trainer.strategy.barrier('fit_done_published')
            # Load weights only for test; keep stopped optimizer/RNG/early/cost state.
            best=torch.load(checkpoint.best_model_path,map_location='cpu',weights_only=False)
            model.load_state_dict(best['state_dict']);del best
            tracking.test_observation_step=checkpoint.best_step
            test=trainer.test(model=model,datamodule=module,ckpt_path=None,verbose=False)
            memory.ensure()
        method_summary=module.method_summary() if hasattr(module,'method_summary') else None
        tracking.snapshot(trainer)
        if rank==0:
            metadata['ranks']=tracking.rank_facts;write_json(out/'launch_metadata.json',metadata)
            summary=dict(status='completed' if reason=='early_stopping' and test else 'stopped',kind=kind,
                stop_reason=reason,best_step=checkpoint.best_step,stop_step=stop_step,
                best_checkpoint=checkpoint.best_model_path or None,final_checkpoint=checkpoint.final_checkpoint,
                best_validation_ndcg10=float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else None,
                test=test,counts=tracking.global_counts,elapsed_seconds=time.perf_counter()-tracking.started,
                ended_unix=time.time(),world_size=4,early_stopping=dict(wait_count=early.wait_count,best_score=metrics_json({'best_score':early.best_score}).get('best_score')),metadata=metadata)
            if method_summary is not None:summary['method']=method_summary
            write_json(out/'summary.json',summary)
        tracking.status(trainer,'completed' if reason=='early_stopping' and test else 'stopped',stop_reason=reason)
    except BaseException as error:
        # Do not enter fresh collectives after an asymmetric rank failure.
        tracking.cpu_snapshot()
        failure=dict(status='failed',kind=kind,stop_reason='error',error=repr(error),rank=rank,pid=os.getpid(),
            global_step=trainer.global_step if trainer else 0,last_collective_counts=tracking.global_counts,
            local_counts=tracking.local,elapsed_seconds=time.perf_counter()-tracking.started,ended_unix=time.time(),metadata=metadata)
        if rank==0:write_json(out/'status.json',failure);write_json(out/'summary.json',failure)
        else:print(json.dumps(failure),file=sys.stderr,flush=True)
        raise

if __name__=='__main__':main()
