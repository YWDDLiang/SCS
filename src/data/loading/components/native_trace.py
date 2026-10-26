"""Compact, atomic records of consumed native microbatches."""
import json
import os
import time
from pathlib import Path
import torch

class TraceExhaustedError(RuntimeError):
    pass

class NativeTraceWriter:
    def __init__(self, directory, metadata, chunk_batches=128):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'metadata.json'
        if self.path.exists():
            raise FileExistsError(f'Trace already exists: {self.path}; use a new run directory')
        self.metadata = dict(metadata, format_version=1, chunks=[], consumed_batches=0, complete=False)
        self.chunk_batches = chunk_batches
        self.pending = []
        self._publish()

    def _publish(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.metadata, indent=2, default=str))
        temporary.replace(self.path)

    def append(self, occurrences):
        self.pending.append(occurrences.detach().cpu().to(torch.int32).clone())
        if len(self.pending) >= self.chunk_batches:
            self.flush()

    def flush(self, complete=False):
        if self.pending:
            name = f'chunk_{len(self.metadata["chunks"]):07d}.pt'
            offsets = torch.tensor([0] + [len(x) for x in self.pending], dtype=torch.int64).cumsum(0)
            target = self.directory / name
            temporary = target.with_suffix('.tmp')
            torch.save({'occurrences': torch.cat(self.pending), 'batch_offsets': offsets}, temporary)
            temporary.replace(target)
            self.metadata['chunks'].append(name)
            self.metadata['consumed_batches'] += len(self.pending)
            self.pending.clear()
        self.metadata['complete'] = complete
        self._publish()

class NativeTraceReader:
    def __init__(self, directory, follow=False, poll_interval=0.5, source_status_path=None):
        if poll_interval <= 0:
            raise ValueError('poll_interval must be positive')
        self.directory = Path(directory)
        self.follow = follow
        self.poll_interval = poll_interval
        self.source_status_path = (Path(source_status_path) if source_status_path
                                   else self.directory.parent / 'status.json')
        self.wait_seconds = 0.0

    def _source_unavailable(self):
        try:
            status = json.loads(self.source_status_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            return f'source status unavailable: {self.source_status_path} ({error})'
        state = str(status.get('status', '')).lower()
        if state in {'completed', 'stopped', 'failed', 'cancelled', 'canceled', 'killed', 'error', 'finished'}:
            return f'source status is terminal: {state}'
        try:
            pid = int(status['pid'])
            if pid <= 0:
                return 'source status has invalid pid'
            if os.name == 'nt':
                import ctypes
                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                if not handle:
                    return f'source process {pid} has exited'
                ctypes.windll.kernel32.CloseHandle(handle)
            else:
                os.kill(pid, 0)
        except ProcessLookupError:
            return f'source process {pid} has exited'
        except PermissionError:
            pass  # A live process may be owned by another account.
        except (KeyError, TypeError, ValueError):
            return 'source status has no valid pid; refusing to wait indefinitely'
        return None

    def batches(self):
        chunk_index = 0
        while True:
            metadata = json.loads((self.directory / 'metadata.json').read_text())
            if chunk_index >= len(metadata['chunks']):
                reason = 'published trace currently exhausted'
                if self.follow and not metadata.get('complete', False):
                    reason = self._source_unavailable()
                    if reason is None:
                        started = time.monotonic()
                        time.sleep(self.poll_interval)
                        self.wait_seconds += time.monotonic() - started
                        continue
                elif metadata.get('complete', False):
                    reason = 'source trace marked complete'
                # Re-read after observing terminal status so a final atomic flush
                # racing the first metadata read is not lost.
                if self.follow:
                    latest = json.loads((self.directory / 'metadata.json').read_text())
                    if len(latest['chunks']) > chunk_index:
                        continue
                    metadata = latest
                raise TraceExhaustedError(f'Native trace exhausted after {metadata["consumed_batches"]} microbatches ({reason}). This is not convergence. Extend the source trace and resume from the saved consumed_batches cursor.')
            payload = torch.load(self.directory / metadata['chunks'][chunk_index], map_location='cpu', weights_only=True)
            offsets = payload['batch_offsets'].tolist()
            for start, end in zip(offsets, offsets[1:]):
                yield payload['occurrences'][start:end]
            chunk_index += 1


def reconstruct_occurrences(store, occurrences, uid_to_row=None):
    if uid_to_row is None:
        uid_to_row = {int(uid): row for row, uid in enumerate(store.user_ids)}
    batch = []
    for uid, start, target in occurrences.tolist():
        row = uid_to_row[uid]
        offset = int(store.offsets[row])
        hierarchy = store.sid_hierarchy
        batch.append({'sequence_data': store.tokens[offset + start * hierarchy:offset + (target + 1) * hierarchy].to(torch.int64), 'user_id': store.user_ids[row].reshape(1).clone()})
    return batch
