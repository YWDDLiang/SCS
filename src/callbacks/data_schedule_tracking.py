"""Run accounting and resumable model RNG, independent of data loader RNG."""
import json
import os
import random
import resource
import psutil
import time
from pathlib import Path
import numpy as np
import torch
from lightning.pytorch import Callback


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def metrics_json(metrics):
    result = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item() if value.numel() == 1 else None
        if isinstance(value, (int, float)):
            result[str(key)] = float(value) if np.isfinite(value) else None
    return result


class DataScheduleTracking(Callback):
    """Synchronized phase wall time (device occupancy), not CUDA kernel busy time.

    Train device seconds measures synchronized train batch hooks; fetch/transfer before
    the start hook is recorded under data_wait_seconds;
    validation/test device seconds include their complete synchronized phase wall time.
    CPU time is the main process only; worker CPU is explicitly not included.
    """
    def __init__(self, output_dir, kind='formal', restart_data_stream=False, memory_reservation=None):
        self.output_dir = Path(output_dir)
        self.memory_reservation = memory_reservation
        self.kind = kind
        self.restart_data_stream = restart_data_stream
        self.counts = dict(microbatches=0, examples=0, nonpadding_tokens=0,
                           train_device_seconds=0., validation_device_seconds=0.,
                           test_device_seconds=0., data_wait_seconds=0., main_process_cpu_seconds=0.)
        self.worker_cpu_prefix = 0.
        self.main_cpu_prefix = 0.
        self.pending_rng = None
        self.saved_step200 = False
        self.started = time.perf_counter()
        self.last_batch_end = None

    def snapshot_worker_cpu(self):
        self.counts['main_process_cpu_seconds'] = self.main_cpu_prefix + time.process_time()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        seconds = usage.ru_utime + usage.ru_stime
        for child in psutil.Process().children(recursive=True):
            try:
                times = child.cpu_times(); seconds += times.user + times.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.counts['worker_cpu_seconds'] = self.worker_cpu_prefix + seconds

    def ensure_memory(self):
        if self.memory_reservation is not None:
            self.memory_reservation.ensure()

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def state_dict(self):
        return {'counts': self.counts, 'saved_step200': self.saved_step200}

    def load_state_dict(self, state_dict):
        self.counts.update(state_dict['counts'])
        self.saved_step200 = state_dict.get('saved_step200', False)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['data_schedule_model_rng'] = dict(python=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.pending_rng = checkpoint.get('data_schedule_model_rng')
        if self.pending_rng is None:
            raise RuntimeError('Checkpoint lacks model RNG: exact continuation cannot be claimed')

    def on_train_start(self, trainer, pl_module):
        self.ensure_memory()
        self.worker_cpu_prefix = self.counts.get('worker_cpu_seconds', 0.)
        self.main_cpu_prefix = self.counts.get('main_process_cpu_seconds', 0.)
        if self.pending_rng:
            state = self.pending_rng
            random.setstate(state['python']); np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch'])
            if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])
            self.pending_rng = None
        self.sync()
        self.last_batch_end = time.perf_counter()
        self.cpu_start = time.process_time()
        self.status(trainer, 'running')

    def status(self, trainer, status, **extra):
        write_json(self.output_dir / 'status.json', dict(status=status, kind=self.kind,
            pid=os.getpid(), global_step=trainer.global_step, counts=self.counts, gpu_memory_reservation=self.memory_reservation.metadata() if self.memory_reservation else None, **extra))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.ensure_memory()
        self.sync()
        now = time.perf_counter()
        if self.last_batch_end is not None:
            self.counts['data_wait_seconds'] += now - self.last_batch_end
        self.batch_start = now
        data = batch
        while isinstance(data, (list, tuple)) and data:
            data = data[0]
        if hasattr(data, 'transformed_sequences'):
            data = data.transformed_sequences
        if isinstance(data, dict):
            tensor = data.get('sequence_data', data.get('input_ids'))
            if isinstance(tensor, torch.Tensor):
                self.counts['examples'] += tensor.shape[0]
                self.counts['nonpadding_tokens'] += int((tensor != -1).sum().item())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.ensure_memory()
        self.sync()
        self.last_batch_end = time.perf_counter()
        self.counts['train_device_seconds'] += self.last_batch_end - self.batch_start
        self.counts['microbatches'] += 1
        if hasattr(trainer.datamodule, 'mark_batch_consumed'):
            trainer.datamodule.mark_batch_consumed(batch)
        self.counts['main_process_cpu_seconds'] += time.process_time() - self.cpu_start
        self.cpu_start = time.process_time()
        if trainer.global_step == 200 and not self.saved_step200:
            self.saved_step200 = True
            trainer.save_checkpoint(str(self.output_dir / 'checkpoints' / 'step200.ckpt'))
        if self.counts['microbatches'] % trainer.accumulate_grad_batches == 0:
            self.status(trainer, 'running')

    def on_validation_start(self, trainer, pl_module):
        self.ensure_memory()
        self.sync(); self.eval_start = time.perf_counter(); self.eval_cpu_start = time.process_time()

    def on_validation_end(self, trainer, pl_module):
        self.ensure_memory()
        self.sync()
        self.counts['validation_device_seconds'] += time.perf_counter() - self.eval_start
        self.counts['main_process_cpu_seconds'] += time.process_time() - self.eval_cpu_start
        self.snapshot_worker_cpu()
        metrics = metrics_json(trainer.callback_metrics)
        checkpoint = self.output_dir / 'checkpoints' / f'eligible-step{trainer.global_step:08d}.ckpt'
        record = dict(step=trainer.global_step, quality=metrics, counts=dict(self.counts),
            elapsed_seconds=time.perf_counter()-self.started, checkpoint=str(checkpoint),
            gpu_device_seconds=sum(self.counts[key] for key in ['train_device_seconds','validation_device_seconds','test_device_seconds']),
            cost_definition='single GPU synchronized phase wall seconds; train excludes loader wait; not kernel busy time',
            cpu_definition='main process CPU seconds plus worker_cpu_seconds (live psutil children plus reaped RUSAGE_CHILDREN), separately recorded')
        self.pending_validation_record = record
        self.last_batch_end = time.perf_counter()
        self.cpu_start = time.process_time()
        self.write_per_user(trainer, pl_module, 'validation')
        self.status(trainer, 'running')

    def write_per_user(self, trainer, pl_module, split):
        user_ids, values = pl_module.evaluator.gathered_per_user_ndcg10()
        folder = self.output_dir / 'per_user' / split
        folder.mkdir(parents=True, exist_ok=True)
        observation_step = getattr(self, 'test_observation_step', trainer.global_step) if split == 'test' else trainer.global_step
        torch.save(dict(format_version=1, split=split, global_step=observation_step, metric='ndcg@10', user_ids=user_ids, values=values, mean=float(values.double().mean())), folder / f'step_{observation_step:08d}.pt')

    def on_test_start(self, trainer, pl_module):
        self.ensure_memory()
        self.sync(); self.eval_start = time.perf_counter(); self.eval_cpu_start = time.process_time()

    def on_test_end(self, trainer, pl_module):
        self.ensure_memory()
        self.sync(); self.counts['test_device_seconds'] += time.perf_counter() - self.eval_start
        self.counts['main_process_cpu_seconds'] += time.process_time() - self.eval_cpu_start
        self.write_per_user(trainer, pl_module, 'test')


from lightning.pytorch.callbacks import ModelCheckpoint


class BranchModelCheckpoint(ModelCheckpoint):
    """Keep the inherited validation best for P4 without deleting prefix assets."""
    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        state['dirpath'] = self.dirpath
        super().load_state_dict(state)

    def _remove_checkpoint(self, trainer, filepath):
        if Path(filepath).resolve().parent == Path(self.dirpath).resolve():
            super()._remove_checkpoint(trainer, filepath)


class TrackingCheckpointFinalizer(ModelCheckpoint):
    """Ordered after the quality checkpoint callback by Lightning's checkpoint ordering."""
    def __init__(self, tracking):
        super().__init__(save_top_k=0, save_last=False)
        self.tracking = tracking

    def on_validation_end(self, trainer, pl_module):
        saving_start = time.perf_counter()
        folder = self.tracking.output_dir / 'checkpoints'
        trainer.save_checkpoint(str(folder / f'eligible-step{trainer.global_step:08d}.ckpt'))
        if trainer.global_step == 200:
            trainer.save_checkpoint(str(folder / 'step200.ckpt'))
        self.tracking.counts['checkpoint_wall_seconds'] = self.tracking.counts.get('checkpoint_wall_seconds', 0.) + time.perf_counter() - saving_start
        self.tracking.snapshot_worker_cpu()
        record = self.tracking.pending_validation_record
        record['counts'] = dict(self.tracking.counts)
        record['elapsed_seconds'] = time.perf_counter() - self.tracking.started
        with (self.tracking.output_dir / 'validation_curve.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        self.tracking.last_batch_end = time.perf_counter()
        self.tracking.cpu_start = time.process_time()

    def on_train_epoch_end(self, trainer, pl_module):
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass
