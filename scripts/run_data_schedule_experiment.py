#!/usr/bin/env python
"""Single device data schedule experiments. GPU allocation belongs to the parent."""
import os
# Only this process's trusted local run checkpoints are loaded by Lightning.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import argparse
import hashlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
torch.set_float32_matmul_precision('medium')
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from src.utils import custom_hydra_resolvers
from src.utils.gpu_memory_reservation import GPUMemoryReservation
from src.callbacks.data_schedule_tracking import DataScheduleTracking, BranchModelCheckpoint, TrackingCheckpointFinalizer, metrics_json, write_json


def build_config(args):
    out = str(Path(args.output_dir).resolve())
    data_root = Path(getattr(args, 'data_root', None) or ROOT / 'data/amazon_data')
    sid_path = getattr(args, 'semantic_id_path', None) or str(ROOT / f'reused-assets/{args.dataset}/semantic_ids.pt')
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=['experiment=tiger_train_flat',
            f'data_dir={data_root}/{args.dataset}',
            f'semantic_id_path={sid_path}', 'num_hierarchies=4',
            f'seed={args.seed}', '+model.top_k_for_generation=50'])
    with open_dict(cfg):
        cfg.paths.output_dir = out
        cfg.paths.work_dir = str(ROOT)
        cfg.paths.profile_dir = out + '/profile'
        cfg.paths.metadata_dir = out + '/metadata'
        cfg.trainer.max_epochs = -1
        cfg.trainer.max_steps = args.pilot_updates or args.max_updates or 320000
        cfg.trainer.devices = 1
        cfg.trainer.strategy = 'auto'
        cfg.trainer.sync_batchnorm = False
        cfg.trainer.accumulate_grad_batches = args.accumulation
        cfg.trainer.val_check_interval = args.validation_updates * args.accumulation
        cfg.trainer.check_val_every_n_epoch = None
        cfg.trainer.precision = '32-true'
        cfg.trainer.enable_progress_bar = False
        cfg.trainer.enable_model_summary = False
        cfg.trainer.num_sanity_val_steps = 0
        if getattr(args, 'deterministic_probe', False):
            cfg.trainer.deterministic = True
        if getattr(args, 'efficient_eval', False):
            cfg.model._target_ = 'src.models.modules.semantic_id.efficient_tiger_generation_model.EfficientSemanticIDEncoderDecoder'
        cfg.data_loading.datamodule = {
            '_target_': 'src.data.loading.datamodules.scheduled_datamodule.ScheduledSequenceDataModule',
            'train_dataloader_config': '${data_loading.train_dataloader_config.dataloader}',
            'val_dataloader_config': '${data_loading.val_dataloader_config.dataloader}',
            'test_dataloader_config': '${data_loading.test_dataloader_config.dataloader}',
            'mode': args.mode, 'candidate_store_path': args.candidate_store or str(ROOT / f'data/subsequence-candidate-stores/{args.dataset}_train.pt'),
            'trace_dir': out + '/trace', 'source_trace_dir': args.source_trace_dir,
            'follow_source_trace': getattr(args, 'follow_source_trace', False),
            'fast_scheduled_collate': getattr(args, 'fast_data', False),
            'native_source_dir': getattr(args, 'native_source_dir', None),
            'loader_context': getattr(args, 'loader_context', None),
            'eval_cache_paths': ({'validate': str(Path(args.eval_cache_dir) / f'{args.dataset}_validation.pt'),
                                  'test': str(Path(args.eval_cache_dir) / f'{args.dataset}_test.pt')}
                                 if getattr(args, 'eval_cache_dir', None) else {}),
            'selected_indices_path': args.selected_indices, 'selection_ratio': args.retention,
            'restart_stream_on_restore': args.restart_data_stream, 'stream_seed': args.seed, 'trace_chunk_batches': 128}
        cfg.eval.evaluator._target_ = 'src.components.joint_scheduler_eval.RecordingSIDRetrievalEvaluator'
        for stage in ['val_dataloader_config', 'test_dataloader_config']:
            cfg.data_loading[stage].dataloader.collate_fn._target_ = 'src.data.loading.components.joint_scheduler_collate.collate_fn_train_with_user_ids'
        if getattr(args, 'loader_timeout', None) is not None:
            for stage in ['train_dataloader_config', 'val_dataloader_config', 'test_dataloader_config']:
                cfg.data_loading[stage].dataloader.timeout = args.loader_timeout
        cfg.callbacks = {}
        cfg.logger = {}
        cfg.ckpt_path = args.checkpoint
    OmegaConf.resolve(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['beauty','toys','sports'], required=True)
    parser.add_argument('--mode', choices=['native_dynamic','native_replay','global_uniform'], required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--retention', type=float, default=1.)
    parser.add_argument('--candidate-store')
    parser.add_argument('--data-root')
    parser.add_argument('--semantic-id-path')
    parser.add_argument('--native-source-dir', help='Persistent CPU native stream; original batches and checkpointable consumer cursor')
    parser.add_argument('--eval-cache-dir', help='Verified original finite validation/test batches')
    parser.add_argument('--loader-context', choices=['spawn', 'fork', 'forkserver'])
    parser.add_argument('--loader-timeout', type=int)
    parser.add_argument('--source-trace-dir')
    parser.add_argument('--follow-source-trace', action='store_true', help='Replay atomically published native chunks while the source is still training; wait for new chunks')
    parser.add_argument('--fast-data', action='store_true', help='Vectorized CPU collation with identical scheduled input tensors and sample order')
    parser.add_argument('--efficient-eval', action='store_true', help='Reuse the identical encoder result within an evaluation batch; training is unchanged')
    parser.add_argument('--selected-indices')
    parser.add_argument('--checkpoint')
    parser.add_argument('--restart-data-stream', action='store_true', help='P4 branch: restore model/optimizer/RNG/early-stop, restart the selected data stream')
    parser.add_argument('--pilot-updates', type=int)
    parser.add_argument('--max-updates', type=int, help='Explicit bounded pilot, never a converged formal result')
    parser.add_argument('--accumulation', type=int, default=128)
    parser.add_argument('--validation-updates', type=int, default=50)
    parser.add_argument('--config-only', action='store_true')
    parser.add_argument('--deterministic-probe', action='store_true', help='Bounded engineering diagnostic only; not a formal training option')
    args = parser.parse_args()
    if not 0 < args.retention <= 1: parser.error('retention must lie in (0,1]')
    if args.mode == 'native_replay' and not args.source_trace_dir: parser.error('native_replay requires --source-trace-dir')
    if args.follow_source_trace and args.mode != 'native_replay': parser.error('--follow-source-trace is only supported for native_replay')
    if args.fast_data and args.mode == 'native_dynamic': parser.error('--fast-data applies to replay/global; native worker collation is unchanged')
    if args.native_source_dir and args.mode != 'native_dynamic': parser.error('--native-source-dir applies to native_dynamic')
    if args.accumulation <= 0 or args.validation_updates <= 0: parser.error('update intervals must be positive')
    if args.deterministic_probe and not (args.pilot_updates or args.max_updates):
        parser.error('--deterministic-probe requires a bounded engineering pilot')
    cfg = build_config(args)
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)
    OmegaConf.save(cfg, out / 'config.yaml')
    kind = 'pilot' if args.pilot_updates or args.max_updates else 'formal'
    metadata = dict(arguments=vars(args), kind=kind, pid=os.getpid(), hostname=socket.gethostname(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), launched_unix=time.time(),
        runtime_callbacks=dict(early_stopping=dict(monitor='val/ndcg@10', mode='max', min_delta=0., patience=10, check_on_train_epoch_end=False), model_checkpoint=dict(monitor='val/ndcg@10', mode='max', save_top_k=1, save_last=True), eligible_checkpoint='every validation', prefix_checkpoint='step200'), checkpoint_loading='trusted local run artifacts, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1', matmul_precision='medium', trace_dir=str(out / 'trace'), cost_scope='fixed semantic ID representation reused; candidate preprocessing/selection must be added separately',
        resume_policy='native_dynamic exact worker RNG recovery unsupported; replay/global restore committed cursor; restart-data-stream explicitly resets data only')
    source_files = ['scripts/run_data_schedule_experiment.py', 'src/callbacks/data_schedule_tracking.py', 'src/data/loading/datamodules/scheduled_datamodule.py', 'src/data/loading/components/native_trace.py', 'src/data/loading/components/collate_functions.py', 'src/data/loading/components/fast_scheduled_collate.py']
    metadata['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}
    for name in ['src/data/loading/components/native_source_stream.py', 'src/data/loading/components/cached_eval_batches.py']:
        metadata['source_sha256'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    trace_dir = str(Path(args.native_source_dir) / 'trace') if args.native_source_dir else str(out / 'trace')
    metadata['trace_dir'] = trace_dir
    if args.mode == 'native_replay':
        source_metadata_path = Path(args.source_trace_dir) / 'metadata.json'
        if source_metadata_path.exists():
            metadata['source_trace_stream_id'] = json.loads(source_metadata_path.read_text()).get('stream_id')
    if args.native_source_dir:
        metadata['resume_policy'] = 'Same persistent CPU source UUID and retained original batches required for exact native continuation; legacy native checkpoints cannot resume exactly'
        metadata['native_consumed_trace_dir'] = str(Path(args.native_source_dir) / 'native_consumed_trace')
        info_path = Path(args.native_source_dir) / 'source.json'
        if info_path.exists(): metadata['native_source_stream_id'] = json.loads(info_path.read_text())['stream_id']
    if args.efficient_eval:
        name = 'src/models/modules/semantic_id/efficient_tiger_generation_model.py'
        metadata['source_sha256'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    write_json(out / 'launch_metadata.json', metadata)
    if args.config_only:
        write_json(out / 'status.json', dict(status='configured', kind=kind)); return
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Parent must explicitly set CUDA_VISIBLE_DEVICES to one allocated GPU')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible GPU required')
    reservation = GPUMemoryReservation()
    try:
        reservation.ensure()
    except BaseException as error:
        metadata['gpu_memory_reservation'] = reservation.metadata()
        write_json(out / 'launch_metadata.json', metadata)
        failure = dict(status='failed', kind=kind, stop_reason='error', error=repr(error), ended_unix=time.time(), trace_dir=str(out / 'trace'), metadata=metadata, global_step=0)
        write_json(out / 'status.json', failure)
        write_json(out / 'summary.json', failure)
        raise
    metadata['gpu_memory_reservation'] = reservation.metadata()
    metadata['device'] = dict(name=torch.cuda.get_device_name(0), uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unknown')))
    write_json(out / 'launch_metadata.json', metadata)
    tracking = DataScheduleTracking(out, kind, args.restart_data_stream, memory_reservation=reservation)
    early = EarlyStopping(monitor='val/ndcg@10', mode='max', patience=10, min_delta=0., check_on_train_epoch_end=False, check_finite=True)
    best = BranchModelCheckpoint(dirpath=out / 'checkpoints', filename='best-step{step:08d}', monitor='val/ndcg@10', mode='max', save_top_k=1, save_last=True, save_on_train_epoch_end=False, auto_insert_metric_name=False)
    trainer = None
    module = None
    try:
        L.seed_everything(args.seed, workers=True)
        # Model initialization precedes data module instantiation to isolate initialization RNG.
        model = hydra.utils.instantiate(cfg.model)
        import random
        import numpy as np
        tracking.pending_rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
        module = hydra.utils.instantiate(cfg.data_loading.datamodule)
        trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[early, tracking, best, TrackingCheckpointFinalizer(tracking)], logger=CSVLogger(str(out), name='csv'))
        trainer.fit(model=model, datamodule=module, ckpt_path=args.checkpoint)
        reservation.ensure()
        pilot_final_checkpoint = None
        if kind == 'pilot':
            pilot_final_checkpoint = str(out / 'checkpoints' / 'pilot_final.ckpt')
            trainer.save_checkpoint(pilot_final_checkpoint)
        quality = metrics_json(trainer.callback_metrics)
        stop_reason = 'pilot' if kind == 'pilot' else ('nonfinite' if quality.get('val/ndcg@10') is None else ('early_stopping' if early.stopped_epoch > 0 or (trainer.should_stop and early.wait_count >= early.patience) else 'max_steps'))
        best_step = None
        if best.best_model_path:
            best_step = torch.load(best.best_model_path, map_location='cpu', weights_only=False)['global_step']
        stop_step = trainer.global_step
        test_metrics = []
        if kind == 'formal' and stop_reason == 'early_stopping':
            if not best.best_model_path: raise RuntimeError('Converged run has no validation-best checkpoint')
            write_json(out / 'fit_done.json', dict(stop_reason=stop_reason, kind=kind, global_step=stop_step,
                consumed_batches=module.consumed_batches,
                native_source_stream_id=metadata.get('native_source_stream_id') or metadata.get('source_trace_stream_id')))
            tracking.test_observation_step = best_step
            best_payload = torch.load(best.best_model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(best_payload['state_dict'])
            del best_payload
            test_metrics = trainer.test(model=model, datamodule=module, ckpt_path=None)
            reservation.ensure()
        tracking.snapshot_worker_cpu()
        metadata['gpu_memory_reservation'] = reservation.metadata()
        write_json(out / 'launch_metadata.json', metadata)
        summary = dict(status='completed' if stop_reason == 'early_stopping' and test_metrics else 'stopped', kind=kind,
            stop_reason=stop_reason, best_step=best_step, stop_step=stop_step, trace_dir=trace_dir,
            best_checkpoint=best.best_model_path, pilot_final_checkpoint=pilot_final_checkpoint, best_validation_ndcg10=float(best.best_model_score) if best.best_model_score is not None else None,
            test=test_metrics, counts=tracking.counts, early_stopping=metrics_json({'best_score':early.best_score,'wait_count':early.wait_count}),
            elapsed_seconds=time.perf_counter()-tracking.started, ended_unix=time.time(),
            source_trace_wait_seconds=getattr(getattr(module, 'trace_reader', None), 'wait_seconds', 0.),
            metadata=metadata)
        write_json(out / 'summary.json', summary)
        tracking.status(trainer, summary['status'], stop_reason=stop_reason)
    except BaseException as error:
        metadata['gpu_memory_reservation'] = reservation.metadata()
        write_json(out / 'launch_metadata.json', metadata)
        tracking.snapshot_worker_cpu()
        payload = dict(status='failed', kind=kind, stop_reason='error', error=repr(error),
            global_step=trainer.global_step if trainer else None, counts=dict(tracking.counts),
            elapsed_seconds=time.perf_counter()-tracking.started, ended_unix=time.time(),
            source_trace_wait_seconds=getattr(getattr(module, 'trace_reader', None), 'wait_seconds', 0.),
            trace_dir=trace_dir, metadata=metadata)
        write_json(out / 'status.json', payload); write_json(out / 'summary.json', payload)
        raise

if __name__ == '__main__':
    main()
