"""Read immutable original native batches published by a persistent CPU source."""
import json
import os
import time
from pathlib import Path
import torch


class NativeSourceUnavailable(RuntimeError):
    pass


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def _source_terminal(directory):
    status = read_json(directory / 'status.json') or read_json(directory / 'source.json', {})
    if status.get('status') in {'failed','completed','stopped','cancelled'}:
        return str(status.get('status'))
    pid = status.get('pid')
    if pid:
        try:
            os.kill(int(pid), 0)
            stat = Path(f'/proc/{pid}/stat')
            if stat.exists():
                fields=stat.read_text().rsplit(')',1)[1].split()
                if fields[0]=='Z': return 'source process became zombie'
                if status.get('start_ticks') and fields[19]!=status['start_ticks']:
                    return 'source process identity changed'
        except ProcessLookupError:
            return 'source process exited'
    return None


def read_native_batches(source_dir, start_batch=0):
    """Yield original (input, labels, int32 occurrences), at committed global cursor.

    A cache file is readable only after its matching compact trace chunk appears
    in metadata.json. No candidate reconstruction or model-side RNG is used.
    Missing retained history and terminal source exhaustion raise explicitly.
    """
    directory = Path(source_dir).resolve()
    cursor = int(start_batch)
    if cursor < 0:
        raise ValueError('start_batch must be nonnegative')
    while True:
        info = read_json(directory / 'source.json')
        metadata = read_json(directory / 'trace/metadata.json', {})
        if info:
            chunk_batches = int(info['chunk_batches'])
            chunk_index, within = divmod(cursor, chunk_batches)
            cache_dir = Path(info['batch_cache_dir']).resolve()
            cache_path = cache_dir / f'batch_{chunk_index:07d}.pt'
            if chunk_index < len(metadata.get('chunks', [])):
                if not cache_path.is_file():
                    raise NativeSourceUnavailable(f'Original batch cache missing for cursor {cursor}: {cache_path}; restore from a retained complete checkpoint boundary')
                payload = torch.load(cache_path, map_location='cpu', weights_only=False)
                if payload['start_batch'] != chunk_index * chunk_batches:
                    raise NativeSourceUnavailable('Native cache global cursor mismatch')
                batches = payload['batches']
                if within >= len(batches):
                    raise NativeSourceUnavailable(f'Native cache ends before requested cursor {cursor}')
                for batch in batches[within:]:
                    if len(batch) != 3 or batch[2].dtype != torch.int32:
                        raise NativeSourceUnavailable('Invalid original native batch payload')
                    yield batch
                    cursor += 1
                continue
        terminal = _source_terminal(directory)
        if terminal or metadata.get('complete', False):
            raise NativeSourceUnavailable(f'Original native source exhausted at global batch {cursor}: {terminal or "source complete"}; EOF is not convergence')
        time.sleep(0.5)
