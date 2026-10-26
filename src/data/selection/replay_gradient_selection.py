"""One step200 P4 raw-gradient selection over literal replay occurrences.

Effective-input keys save forward work only. References, self exclusion, pruning
counts, and tie order all refer to original occurrence identities.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
import torch.distributed as dist

from src.data.selection.dynamic_utility import actual_heads


REFERENCE_COUNT = 1024
REFERENCE_SEED = 42 ^ 0x51A7  # Separate stream: seed42 itself would select P[:1024].
FORWARD_BATCH_SIZE = 512
SCORE_CHUNK_SIZE = 1024
SCAN_CHUNK_SIZE = 1_000_000


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def preserved_scoring_state(model):
    """Preserve rank-local Python/NumPy/Torch/CUDA RNG and every module mode."""
    device = next(model.parameters()).device
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda_state = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    modes = [(module, module.training, getattr(module, 'is_training', None))
             for module in model.modules()]
    try:
        model.eval()
        for module, _, special in modes:
            if special is not None:
                module.is_training = False
        yield
    finally:
        for module, training, special in modes:
            module.training = training
            if special is not None:
                module.is_training = special
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


def raw_gradient_factors(model, batch):
    """Original P4 raw softmax-minus-one-hot factors, quantized to fp16.

    This deliberately does not call the dynamic Adam/mean-CE utility scorer.
    Distinct output heads mean that the four independent products are summed.
    """
    model_input, labels = batch
    device = next(model.parameters()).device
    if len(labels.labels) != 1:
        raise ValueError('Exactly one target SID field required')
    mask = model_input.mask.to(device)
    targets = next(iter(labels.labels.values())).reshape(mask.shape[0], -1).to(device).long()
    heads = actual_heads(model)
    if len(heads) != 4 or targets.shape[1] != 4 or bool((targets < 0).any()):
        raise ValueError('P4 replay scoring requires four complete target SID levels')
    fields = {model.feature_to_model_input_map.get(key, key): value.to(device)
              for key, value in model_input.transformed_sequences.items()}
    input_ids = fields.get('input_ids')
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[1] != 120:
        raise ValueError('Effective-input cache is audited for fixed encoder120 only')
    hidden = model.forward(attention_mask_encoder=mask, future_ids=targets, **fields)[:, :-1]
    if hidden.shape[1] != 4:
        raise ValueError('Hidden states do not match the four independent SID heads')
    logits = torch.stack([head(hidden[:, level]) for level, head in enumerate(heads)], dim=1)
    errors = torch.softmax(logits.float(), dim=-1)
    errors.scatter_add_(2, targets.unsqueeze(-1),
                        -torch.ones_like(targets, dtype=errors.dtype).unsqueeze(-1))
    return errors.half(), hidden.half()


def factor_inner_products(errors, hidden, reference_errors, reference_hidden):
    """Exactly the P4 fp16-factor/fp32-dot/sum-four-heads arithmetic."""
    similarity = torch.zeros((len(errors), len(reference_errors)), dtype=torch.float32,
                             device=errors.device)
    for level in range(4):
        term_e = errors[:, level].float() @ reference_errors[:, level].float().T
        term_h = hidden[:, level].float() @ reference_hidden[:, level].float().T
        similarity += term_e * term_h
    return similarity


def occurrence_threshold(scores, counts, reference_keys, exception_scores, dropped):
    """Weighted unique-key threshold, with the exact reference-ID exceptions."""
    counts = np.asarray(counts, dtype=np.int64).copy()
    reference_keys = np.asarray(reference_keys, dtype=np.int64)
    exception_scores = np.asarray(exception_scores, dtype=np.float32)
    np.subtract.at(counts, reference_keys, 1)
    if np.any(counts < 0) or len(reference_keys) != len(exception_scores):
        raise ValueError('Invalid reference occurrence counts')
    active = counts > 0
    values = np.concatenate((np.asarray(scores, dtype=np.float32)[active], exception_scores))
    weights = np.concatenate((counts[active], np.ones(len(exception_scores), dtype=np.int64)))
    if not np.isfinite(values).all() or not 0 < dropped < int(weights.sum()):
        raise ValueError('Nonfinite scores or invalid pruning budget')
    # This sorts about one million computation keys, never the 95m occurrences.
    order = np.argsort(-values, kind='stable')
    cumulative = np.cumsum(weights[order], dtype=np.int64)
    threshold = values[order[np.searchsorted(cumulative, dropped, side='left')]]
    greater = int(weights[values > threshold].sum())
    equal = int(weights[values == threshold].sum())
    tie_drops = dropped - greater
    if not 0 < tie_drops <= equal:
        raise AssertionError('Weighted threshold cannot realize exact occurrence budget')
    return np.float32(threshold), tie_drops, greater, equal


def write_selected_positions(path, permutation, occurrence_to_key, key_scores, key_counts,
                             reference_ids, exception_scores, *, chunk_size=SCAN_CHUNK_SIZE):
    """Delete highest20%, ties by earliest P position; output sorted positions."""
    count = len(permutation)
    if chunk_size <= 0:
        raise ValueError('Selection scan chunk size must be positive')
    budget = 4 * count // 5
    dropped = count - budget
    reference_ids = np.asarray(reference_ids, dtype=np.int64)
    if len(reference_ids) and (np.any(np.diff(reference_ids) <= 0)
                               or reference_ids[0] < 0 or reference_ids[-1] >= count):
        raise ValueError('Reference occurrence IDs must be sorted and distinct')
    reference_keys = np.asarray(occurrence_to_key[reference_ids], dtype=np.int64)
    threshold, remaining_ties, greater, equal = occurrence_threshold(
        key_scores, key_counts, reference_keys, exception_scores, dropped)
    total_tie_drops = remaining_ties
    output = np.lib.format.open_memmap(path, mode='w+', dtype=np.int64, shape=(budget,))
    written = 0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        ids = np.asarray(permutation[start:stop], dtype=np.int64)
        values = np.asarray(key_scores[occurrence_to_key[ids]], dtype=np.float32).copy()
        if len(reference_ids):
            locations = np.searchsorted(reference_ids, ids)
            inside = locations < len(reference_ids)
            inside &= reference_ids[locations.clip(max=len(reference_ids) - 1)] == ids
            values[inside] = exception_scores[locations[inside]]
        keep = values <= threshold
        if remaining_ties:
            tied = np.flatnonzero(values == threshold)
            remove = min(remaining_ties, len(tied))
            keep[tied[:remove]] = False
            remaining_ties -= remove
        selected = np.flatnonzero(keep).astype(np.int64) + start
        if written + len(selected) > budget:
            raise AssertionError('Selection exceeds exact80% occurrence budget')
        output[written:written + len(selected)] = selected
        written += len(selected)
    if written != budget or remaining_ties:
        raise AssertionError('Selection does not realize exact80% occurrence budget')
    output.flush()
    del output
    return dict(selected_count=budget, dropped_count=dropped, threshold=float(threshold),
                strictly_greater_count=greater, threshold_tied_count=equal,
                threshold_ties_deleted=total_tie_drops,
                tie_rule='Delete earlier positions in the fixed original permutation first',
                selected_positions_sorted=True, selection_uses_occurrence_multiplicity=True)


def _load_index(datamodule):
    path = Path(os.environ.get('REPLAY_GRADIENT_INDEX', str(datamodule.assets_path / 'gradient_index')))
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('ready') is not True or manifest.get('effective_history_items') != 29
            or manifest.get('source_manifest_sha256') != file_sha256(datamodule.assets_path / 'manifest.json')
            or manifest.get('source_asset_identity') != datamodule.asset_manifest['immutable_identity']):
        raise RuntimeError('Prepared gradient index does not belong to these literal replay assets')
    for name, facts in manifest['files'].items():
        if file_sha256(path / name) != facts['sha256']:
            raise RuntimeError('Gradient index checksum differs: ' + name)
    mapping = np.load(path / 'occurrence_to_key.npy', mmap_mode='r', allow_pickle=False)
    triples = np.load(path / 'unique_effective_occurrences.npy', mmap_mode='r', allow_pickle=False)
    counts = np.load(path / 'key_counts.npy', mmap_mode='r', allow_pickle=False)
    count, unique_count = len(datamodule.occurrences), len(triples)
    if (mapping.dtype != np.uint32 or mapping.shape != (count,) or triples.dtype != np.int32
            or triples.shape != (unique_count, 3) or counts.dtype != np.int64
            or counts.shape != (unique_count,) or int(counts.sum()) != count
            or np.any(counts <= 0) or manifest['total_occurrences'] != count
            or manifest['unique_effective_inputs'] != unique_count):
        raise RuntimeError('Invalid computational cache map or key counts')
    if datamodule.store.sequence_length != 120 or datamodule.store.sid_hierarchy != 4:
        raise RuntimeError('Effective key clamping requires encoder120 and SID hierarchy4')
    return mapping, triples, counts, manifest, file_sha256(manifest_path)


def _batch(datamodule, triples, indices):
    occurrences = torch.from_numpy(np.asarray(triples[indices], dtype=np.int32).copy())
    return datamodule._fast_collate(occurrences)


def fixed_batch_factors(datamodule, model, triples, indices):
    """Use the same512 forward geometry; padding has zero selection weight."""
    indices = np.asarray(indices, dtype=np.int64)
    if not 0 < len(indices) <= FORWARD_BATCH_SIZE:
        raise ValueError('Factor forward requires one through512 actual input keys')
    padded = np.r_[indices, np.repeat(indices[0], FORWARD_BATCH_SIZE - len(indices))]
    errors, hidden = raw_gradient_factors(model, _batch(datamodule, triples, padded))
    return errors[:len(indices)], hidden[:len(indices)]


def _assert_batch_equal(left, right):
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.dtype != right.dtype or not torch.equal(left, right):
            raise AssertionError('Effective cache and literal occurrence collator outputs differ')
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError('Collator output fields differ')
        for key in left:
            _assert_batch_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError('Collator output lengths differ')
        for a, b in zip(left, right):
            _assert_batch_equal(a, b)
    elif hasattr(left, '__dict__'):
        _assert_batch_equal(vars(left), vars(right))
    elif left != right:
        raise AssertionError('Collator output values differ')


def _equivalence(left, right):
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    passed = finite and torch.allclose(left, right, rtol=1e-6, atol=1e-7)
    result = dict(passed=passed, exact=torch.equal(left, right), rtol=1e-6, atol=1e-7,
                  max_absolute_error=float((left.float() - right.float()).abs().max()))
    if not passed:
        raise AssertionError('Direct-occurrence/cache equivalence failed: ' + json.dumps(result))
    return result


def _actual_cache_audit(datamodule, model, triples, mapping, counts, reference_ids,
                        reference_errors, reference_hidden):
    """Small real GPU check, including a duplicate input and occurrence self-mask."""
    # Find a distinct actual identity sharing a sampled reference input. This
    # scans the already prepared map; it never invents a candidate or sample.
    chosen = reference_ids[counts[mapping[reference_ids]] > 1]
    if not len(chosen):
        raise RuntimeError('Actual duplicate-input audit needs a repeated reference effective key')
    original = int(chosen[0])
    key = int(mapping[original])
    duplicate = None
    for start in range(0, len(mapping), SCAN_CHUNK_SIZE):
        matches = np.flatnonzero(mapping[start:start + SCAN_CHUNK_SIZE] == key) + start
        matches = matches[matches != original]
        if len(matches):
            duplicate = int(matches[0])
            break
    if duplicate is None:
        raise AssertionError('Positive duplicate count has no distinct literal occurrence')
    ids = np.concatenate((reference_ids[:30], np.asarray([original, duplicate], dtype=np.int64)))
    keys = np.asarray(mapping[ids], dtype=np.int64)
    unique = np.unique(keys)
    direct_batch = _batch(datamodule, datamodule.occurrences, ids)
    expected_batch = _batch(datamodule, triples, keys)
    _assert_batch_equal(direct_batch, expected_batch)
    # Compare the production factors already used for reference scoring to
    # literal repeated occurrences, with the same512-row forward geometry.
    ref_keys, first_columns = np.unique(mapping[reference_ids], return_index=True)
    locations = np.searchsorted(ref_keys, keys)
    if np.any(locations >= len(ref_keys)) or not np.array_equal(ref_keys[locations], keys):
        raise AssertionError('Audit occurrence input is absent from the production reference cache')
    columns = torch.as_tensor(first_columns[locations], device=reference_errors.device)
    cache_e, cache_h = reference_errors[columns], reference_hidden[columns]
    direct_e, direct_h = fixed_batch_factors(datamodule, model, datamodule.occurrences, ids)
    factors = dict(errors=_equivalence(cache_e, direct_e), hidden=_equivalence(cache_h, direct_h))
    direct = factor_inner_products(direct_e, direct_h, reference_errors, reference_hidden)
    cached = factor_inner_products(cache_e, cache_h, reference_errors, reference_hidden)
    locations = np.searchsorted(reference_ids, ids)
    inside = locations < len(reference_ids)
    inside &= reference_ids[locations.clip(max=len(reference_ids) - 1)] == ids
    row = torch.as_tensor(np.flatnonzero(inside), device=direct.device)
    column = torch.as_tensor(locations[inside], device=direct.device)
    direct[row, column] = float('-inf')
    cached[row, column] = float('-inf')
    scores = _equivalence(cached.amax(1), direct.amax(1))
    return dict(passed=True, actual_occurrence_count=len(ids), distinct_effective_keys=len(unique),
                forward_batch_rows=FORWARD_BATCH_SIZE, forward_padding_rows=FORWARD_BATCH_SIZE - len(ids),
                cached_factors_from_production_references=True,
                collator_all_fields_exact=True, factors=factors, redundancy=scores,
                duplicate_occurrence_ids=[original, duplicate], duplicate_effective_key=key,
                own_reference_column_excluded_only_for_identical_occurrence=True,
                tolerance_inherited_from_original_P4_audit=True)


def ensure_selection(datamodule):
    """All four ranks call once, before materializing the first post200 batch.

    Returns the selection DIRECTORY containing selected_manifest.json and the
    sorted int64 selected_positions.npy in the original fixed permutation.
    The caller owns post-selection cursor handling and loading that array.
    """
    trainer = datamodule.trainer
    if int(trainer.global_step) != 200:
        raise RuntimeError('Gradient80 activates exactly at global optimizer step200')
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() != 4:
        raise RuntimeError('Inline gradient selection requires the original four-rank run')
    rank, world = dist.get_rank(), dist.get_world_size()
    model = trainer.lightning_module  # Never call the DDP wrapper or a training step.
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise RuntimeError('Formal four-rank selector expects a rank-local CUDA device')
    output_dir = Path(datamodule.selection_output_dir)
    selected_path = output_dir / 'selected_positions.npy'
    manifest_path = output_dir / 'selected_manifest.json'
    if manifest_path.exists() or selected_path.exists():
        raise FileExistsError('Existing selection is immutable; load it through datamodule restore')
    torch.cuda.synchronize(device)
    started, cpu_started = time.perf_counter(), time.process_time()
    rank_cost = dict(rank=rank, reference_forward_wall_seconds=0., query_wall_seconds=0.,
                     audit_wall_seconds=0., collective_wait_seconds=0., cpu_selection_wall_seconds=0.,
                     reference_forward_rows=0, reference_padding_rows=0,
                     query_forward_rows=0, query_padding_rows=0,
                     audit_forward_rows=FORWARD_BATCH_SIZE)
    def sync():
        torch.cuda.synchronize(device)
    def collective(tensor):
        sync()
        tick = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        sync()
        rank_cost['collective_wait_seconds'] += time.perf_counter() - tick

    with preserved_scoring_state(model), torch.no_grad():
        mapping, triples, counts, index_manifest, index_manifest_sha = _load_index(datamodule)
        occurrence_count, unique_count = len(mapping), len(triples)
        # Same uniform-without-replacement P4 sampler, domain-separated from P.
        if rank == 0:
            generator = torch.Generator(device='cpu').manual_seed(REFERENCE_SEED)
            reference = torch.randperm(occurrence_count, generator=generator,
                                       dtype=torch.int64, device='cpu')[:REFERENCE_COUNT].sort().values
            reference_device = reference.to(device)
        else:
            reference_device = torch.empty(REFERENCE_COUNT, dtype=torch.int64, device=device)
        dist.broadcast(reference_device, src=0)
        reference_ids = reference_device.cpu().numpy().copy()
        reference_keys = np.asarray(mapping[reference_ids], dtype=np.int64)
        reference_unique, reference_inverse = np.unique(reference_keys, return_inverse=True)
        heads = actual_heads(model)
        levels, vocab, hidden_dim = 4, heads[0].weight.shape[0], heads[0].weight.shape[1]
        if any(head.weight.shape != heads[0].weight.shape for head in heads):
            raise RuntimeError('P4 factor cache requires equal independent-head shapes')
        ref_e = torch.zeros((len(reference_unique), levels, vocab), dtype=torch.float16, device=device)
        ref_h = torch.zeros((len(reference_unique), levels, hidden_dim), dtype=torch.float16, device=device)
        ref_coverage = torch.zeros(len(reference_unique), dtype=torch.int32, device=device)
        owned = np.array_split(np.arange(len(reference_unique), dtype=np.int64), world)[rank]
        local_scores = np.zeros(unique_count, dtype=np.float32)
        local_coverage = np.zeros(unique_count, dtype=np.int32)
        tick = time.perf_counter()
        for start in range(0, len(owned), FORWARD_BATCH_SIZE):
            positions = owned[start:start + FORWARD_BATCH_SIZE]
            ids = reference_unique[positions]
            errors, hidden = fixed_batch_factors(datamodule, model, triples, ids)
            rank_cost['reference_forward_rows'] += FORWARD_BATCH_SIZE
            rank_cost['reference_padding_rows'] += FORWARD_BATCH_SIZE - len(ids)
            positions_device = torch.as_tensor(positions, device=device)
            ref_e[positions_device], ref_h[positions_device] = errors, hidden
            ref_coverage[positions_device] = 1
        sync()
        rank_cost['reference_forward_wall_seconds'] = time.perf_counter() - tick
        collective(ref_e)
        collective(ref_h)
        collective(ref_coverage)
        if not bool((ref_coverage == 1).all()):
            raise RuntimeError('Reference effective-key factors do not have exactly one owner')
        inverse_device = torch.as_tensor(reference_inverse, device=device)
        reference_errors, reference_hidden = ref_e[inverse_device], ref_h[inverse_device]
        # Repeated reference inputs remain repeated columns. Only the exact
        # original occurrence self-column is removed for these 1024 exceptions.
        tick = time.perf_counter()
        reference_similarity = factor_inner_products(ref_e, ref_h, reference_errors, reference_hidden)
        for start in range(0, len(owned), SCORE_CHUNK_SIZE):
            positions = owned[start:start + SCORE_CHUNK_SIZE]
            local_scores[reference_unique[positions]] = reference_similarity[positions].amax(1).cpu().numpy()
            local_coverage[reference_unique[positions]] = 1
        exception_similarity = reference_similarity[inverse_device].clone()
        exception_similarity[torch.arange(REFERENCE_COUNT, device=device),
                             torch.arange(REFERENCE_COUNT, device=device)] = float('-inf')
        exception_scores = exception_similarity.amax(1).cpu().numpy().copy()
        if not np.isfinite(exception_scores).all():
            raise RuntimeError('Nonfinite self-excluded reference occurrence scores')
        audit = _actual_cache_audit(datamodule, model, triples, mapping, counts, reference_ids,
                                    reference_errors, reference_hidden)
        sync()
        rank_cost['audit_wall_seconds'] = time.perf_counter() - tick
        del reference_similarity, exception_similarity, ref_e, ref_h, ref_coverage
        tick = time.perf_counter()
        owned_query_keys = 0
        for start in range(rank * FORWARD_BATCH_SIZE, unique_count, world * FORWARD_BATCH_SIZE):
            ids = np.arange(start, min(start + FORWARD_BATCH_SIZE, unique_count), dtype=np.int64)
            positions = np.searchsorted(reference_unique, ids)
            is_reference = positions < len(reference_unique)
            is_reference &= reference_unique[positions.clip(max=len(reference_unique) - 1)] == ids
            ids = ids[~is_reference]
            if not len(ids):
                continue
            errors, hidden = fixed_batch_factors(datamodule, model, triples, ids)
            rank_cost['query_forward_rows'] += FORWARD_BATCH_SIZE
            rank_cost['query_padding_rows'] += FORWARD_BATCH_SIZE - len(ids)
            similarity = factor_inner_products(errors, hidden, reference_errors, reference_hidden)
            local_scores[ids] = similarity.amax(1).cpu().numpy()
            local_coverage[ids] = 1
            owned_query_keys += len(ids)
            del errors, hidden, similarity
        sync()
        rank_cost['query_wall_seconds'] = time.perf_counter() - tick
        for start in range(0, unique_count, 65_536):
            stop = min(start + 65_536, unique_count)
            piece = torch.from_numpy(local_scores[start:stop].copy()).to(device)
            coverage = torch.from_numpy(local_coverage[start:stop].copy()).to(device)
            collective(piece)
            collective(coverage)
            local_scores[start:stop] = piece.cpu().numpy()
            local_coverage[start:stop] = coverage.cpu().numpy()
        if not np.all(local_coverage == 1) or not np.isfinite(local_scores).all():
            raise RuntimeError('Unique-input scoring has missing, overlapping, or nonfinite results')
        # Scores and identities must agree across all ranks before CPU pruning.
        facts = dict(reference_sha256=array_sha256(reference_ids), score_sha256=array_sha256(local_scores),
                     exception_sha256=array_sha256(exception_scores), audit=audit)
        gathered = [None] * world
        dist.all_gather_object(gathered, facts)
        for row in gathered:
            if any(row[name] != facts[name] for name in ('reference_sha256', 'score_sha256', 'exception_sha256')):
                raise RuntimeError('Ranks disagree about reference identities or redundancy scores')
        pending = output_dir / ('.gradient_selection.pending_' + uuid.uuid4().hex)
        selected_facts = None
        tick = time.perf_counter()
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            pending.mkdir()
            np.save(pending / 'reference_occurrence_ids.npy', reference_ids, allow_pickle=False)
            np.save(pending / 'reference_exception_scores.npy', exception_scores, allow_pickle=False)
            np.save(pending / 'unique_scores.npy', local_scores, allow_pickle=False)
            selected_facts = write_selected_positions(pending / 'selected_positions.npy',
                datamodule.permutation, mapping, local_scores, counts, reference_ids, exception_scores)
            if len(np.load(pending / 'selected_positions.npy', mmap_mode='r')) != 4 * occurrence_count // 5:
                raise AssertionError('Published selected positions have the wrong count')
            file_specs = {'selected_positions.npy': ('int64', [4 * occurrence_count // 5]),
                          'reference_occurrence_ids.npy': ('int64', [REFERENCE_COUNT]),
                          'reference_exception_scores.npy': ('float32', [REFERENCE_COUNT]),
                          'unique_scores.npy': ('float32', [unique_count])}
            files = {name: dict(path=str(output_dir / name), sha256=file_sha256(pending / name),
                               dtype=dtype, shape=shape, bytes=(pending / name).stat().st_size)
                     for name, (dtype, shape) in file_specs.items()}
        sync()
        dist.barrier()
        sync()
        rank_cost['cpu_selection_wall_seconds'] = time.perf_counter() - tick
        rank_cost.update(scoring_wall_seconds=time.perf_counter() - started,
                         cpu_seconds=time.process_time() - cpu_started,
                         unique_query_keys_owned=owned_query_keys,
                         unique_reference_keys_owned=len(owned))
        rank_costs = [None] * world
        dist.all_gather_object(rank_costs, rank_cost)
        if rank == 0:
            identity = dict(source_asset_identity=datamodule.pool_sha256,
                            index_identity=index_manifest['immutable_identity'], step=200,
                            files={name: value['sha256'] for name, value in files.items()})
            selection_manifest = dict(format_version=1, ready=True, method='P4 raw output-head gradient redundancy80',
                source_asset_identity=datamodule.pool_sha256,
                replay_shuffle_asset_identity=datamodule.asset_manifest['immutable_identity'],
                source_manifest_sha256=file_sha256(datamodule.assets_path / 'manifest.json'),
                source_stream_id=datamodule.asset_manifest['source_stream_id'],
                permutation_sha256=datamodule.asset_file_hashes['permutation.npy'],
                immutable_identity=hashlib.sha256(json.dumps(identity, sort_keys=True,
                                                              separators=(',', ':')).encode()).hexdigest(),
                step=200, total_occurrences=occurrence_count, occurrence_count=occurrence_count,
                selected_count=4 * occurrence_count // 5, selection_ratio=.8,
                seed=42, model_matmul_precision=torch.get_float32_matmul_precision(),
                unique_effective_inputs=unique_count, computational_cache_only=True,
                forward_geometry=dict(fixed_batch_size=FORWARD_BATCH_SIZE,
                    references='Distinct reference effective keys split contiguously across four ranks; shorter batches padded to512',
                    queries='Fixed blocks of512 unique keys, omitting already cached reference keys; shorter batches padded to512',
                    reference_key_scores='Computed from the already cached reference-key factors; not forwarded again',
                    padding='Copies of an existing batch row, trimmed immediately; no reference, membership, score or selection weight; compute cost included',
                    direct_cache_audit='Production reference cached factors compared against32 actual literal occurrence rows in a padded512 forward, including distinct duplicate input identities'),
                gradient='Raw softmax-minus-label; independent four output heads; no mean rescaling',
                factor_dtype='fp16', inner_product_dtype='fp32',
                gradient_dimension=levels * vocab * hidden_dim,
                score='Maximum sum of four head inner products against all sampled reference occurrence columns',
                self_exclusion='Only same literal occurrence identity; equivalent-input references remain valid',
                reference_count=REFERENCE_COUNT, reference_seed=REFERENCE_SEED,
                reference_algorithm='torch.randperm(N, private CPU Generator seed42 XOR 0x51A7)[:1024], then sort; independent of shuffle P',
                reference_occurrence_sha256=facts['reference_sha256'],
                reference_distinct_effective_inputs=len(reference_unique),
                index_manifest_sha256=index_manifest_sha, index_identity=index_manifest['immutable_identity'],
                selection=selected_facts, direct_cache_audits=[row['audit'] for row in gathered],
                files=files, rng_and_training_modes_restored=True,
                optimizer_accessed=False, backward_called=False, refreshes=0,
                world_size=world, rank_costs=rank_costs,
                scoring_wall_seconds=time.perf_counter() - started,
                gpu_phase_device_seconds=sum(row['scoring_wall_seconds'] for row in rank_costs),
                cost_definition='Scoring phase wall seconds summed across ranks; includes CPU input, collectives, pruning and peer waiting',
                created_unix=time.time())
            for name in files:
                os.rename(pending / name, output_dir / name)
            _write_json(manifest_path, selection_manifest)  # Ready publication is last.
            pending.rmdir()
        dist.barrier()
        sync()
    # Context exit has restored model modes and the exact incoming RNG state.
    seconds = time.perf_counter() - started
    datamodule.selection_scoring_metadata = dict(manifest=str(manifest_path),
                                                local_scoring_wall_seconds=seconds)
    for callback in trainer.callbacks:
        local = getattr(callback, 'local', None)
        if isinstance(local, dict) and 'train_device_seconds' in local:
            local['score_device_seconds'] = local.get('score_device_seconds', 0.) + seconds
            global_counts = getattr(callback, 'global_counts', None)
            if isinstance(global_counts, dict):
                global_counts.setdefault('score_device_seconds', 0.)
            callback.last_batch_end = time.perf_counter()
    return output_dir
