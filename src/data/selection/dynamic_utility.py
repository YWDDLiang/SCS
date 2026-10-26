"""Read-only utility for the model's distinct SID output heads.

This is a first-order output-head-local approximation, not full Adam influence.
Each per-example CE gradient is averaged over valid target SID tokens. References
are uniformly sampled TRAIN users, one last training target each. Independent
head parameter blocks have no cross-block products. Any future shared-head model
must use a separately audited implementation (it would need cross-position terms).
"""
from contextlib import contextmanager
import hashlib
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


@contextmanager
def preserved_scoring_state(model):
    """Isolate Python/NumPy/Torch and this rank's CUDA RNG and module modes."""
    parameter = next(model.parameters())
    device = parameter.device
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    modes = [(module, module.training, getattr(module, 'is_training', None)) for module in model.modules()]
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
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)


def actual_heads(model):
    heads = list(model.decoder.decoder_mlp)
    if not heads or any(not isinstance(head, torch.nn.Linear) or head.bias is not None for head in heads):
        raise ValueError('Scorer requires bias-free Linear decoder_mlp heads')
    if len({id(head.weight) for head in heads}) != len(heads):
        raise ValueError('Shared output weights require cross-position terms; distinct-head scorer refuses them')
    if any(head.weight.ndim != 2 for head in heads):
        raise ValueError('Invalid output-head parameter shape')
    return heads


def forward_factors(model, batch):
    """Teacher-forced model forward without per-candidate gradient materialization."""
    model_input, labels = batch
    device = next(model.parameters()).device
    if len(labels.labels) != 1:
        raise ValueError('Exactly one SID label field required')
    mask = model_input.mask.to(device)
    targets = next(iter(labels.labels.values())).to(device).reshape(mask.shape[0], -1).long()
    valid = targets >= 0
    count = valid.sum(1)
    if bool((count == 0).any()):
        raise ValueError('Every candidate needs a valid target SID')
    # The model's embedding path requires complete target SIDs; ordinary candidate
    # collation supplies all levels. A masked target cannot silently be embedded.
    if not bool(valid.all()):
        raise ValueError('Model forward requires all target SID levels valid')
    fields = {model.feature_to_model_input_map.get(key, key): value.to(device)
              for key, value in model_input.transformed_sequences.items()}
    hidden = model.forward(attention_mask_encoder=mask, future_ids=targets, **fields)[:, :-1]
    heads = actual_heads(model)
    if hidden.shape[1] != len(heads) or targets.shape[1] != len(heads):
        raise ValueError('Hidden/target/head SID hierarchy mismatch')
    logits = torch.stack([head(hidden[:, level]) for level, head in enumerate(heads)], dim=1).float()
    token_losses = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none').reshape_as(targets)
    losses = (token_losses * valid).sum(1) / count
    # Match CE's log_softmax backward numerics while computing the same p-y.
    residual = F.log_softmax(logits, dim=-1).exp()
    residual.scatter_add_(2, targets.unsqueeze(-1), -torch.ones_like(targets, dtype=residual.dtype).unsqueeze(-1))
    residual = residual * (valid / count.unsqueeze(1)).unsqueeze(-1)
    # The active eager T5 path computes the full fixed padded [B,120] encoder
    # shape (+ a fixed optional user slot) and four decoder targets. Padding masks
    # do not unpad the GEMMs. Nonpadding length is descriptive, not compute cost.
    input_ids = fields.get('input_ids')
    if input_ids is None:
        raise ValueError('Candidate input_ids missing')
    if input_ids.ndim!=2 or input_ids.shape[1]!=120 or targets.shape[1]!=4:
        raise ValueError('Constant slot cost audited only for fixed encoder120 and four SID targets')
    cost = torch.ones_like(losses)
    return hidden.float(), residual, losses, cost


def reference_gradient_sums(hidden, residual):
    return [residual[:, level].T @ hidden[:, level] for level in range(hidden.shape[1])]


def contract_alignment(hidden, residual, preconditioned_reference):
    """Inner product with concatenated actual independent-head gradients."""
    alignment = torch.zeros(hidden.shape[0], dtype=torch.float32, device=hidden.device)
    for level, reference in enumerate(preconditioned_reference):
        alignment += ((hidden[:, level] @ reference.T) * residual[:, level]).sum(1)
    return alignment


def adam_output_preconditioner(model, optimizer):
    """Only D_t=1/(sqrt(v_t/(1-beta2**t))+eps); no momentum approximation."""
    if hasattr(optimizer, 'optimizer'):
        optimizer = optimizer.optimizer
    heads = actual_heads(model)
    groups = {id(parameter): group for group in optimizer.param_groups for parameter in group['params']}
    matrices, facts = [], []
    for level, head in enumerate(heads):
        parameter = head.weight
        group = groups.get(id(parameter))
        state = optimizer.state.get(parameter, {})
        if group is None or 'exp_avg_sq' not in state or 'step' not in state or 'betas' not in group:
            raise ValueError(f'Output head {level} lacks trained Adam second moment')
        step = int(state['step'])
        beta2, eps = float(group['betas'][1]), float(group['eps'])
        if step <= 0 or not 0 <= beta2 < 1 or eps <= 0:
            raise ValueError('Invalid Adam bias correction inputs')
        variance = state['exp_avg_sq'].detach().float()
        if variance.shape != parameter.shape or not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
            raise ValueError('Invalid Adam output-head variance')
        matrix = (torch.sqrt(variance / (1. - beta2 ** step)) + eps).reciprocal()
        matrices.append(matrix)
        facts.append(dict(level=level, shape=list(parameter.shape), adam_step=step, beta2=beta2, eps=eps))
    return matrices, facts


def _group_robust_z(values, inverse, counts):
    """Median/MAD within nonempty groups; RMS fallback if MAD is zero."""
    order = np.lexsort((values, inverse))
    sorted_values = values[order]
    starts = np.cumsum(counts) - counts
    median = (sorted_values[starts + (counts - 1) // 2] + sorted_values[starts + counts // 2]) / 2.
    centered = values - median[inverse]
    deviations = np.abs(centered)
    deviation_order = np.lexsort((deviations, inverse))
    sorted_deviations = deviations[deviation_order]
    mad = (sorted_deviations[starts + (counts - 1) // 2] + sorted_deviations[starts + counts // 2]) / 2.
    scale = 1.4826 * mad
    fallback = np.sqrt(np.bincount(inverse, weights=centered * centered, minlength=len(counts)) / counts)
    scale = np.where(scale > 0., scale, fallback)
    scale = np.where(scale > 0., scale, 1.)
    return centered / scale[inverse], int((mad == 0.).sum())


def grouped_utility_weights(alignment, progress, group_ids):
    """CPU robust group utility, temperature=1, progress coefficient=.25."""
    alignment = torch.as_tensor(alignment).detach().cpu().double().numpy()
    progress = torch.as_tensor(progress).detach().cpu().double().numpy()
    groups = torch.as_tensor(group_ids).detach().cpu().long().numpy()
    if alignment.ndim != 1 or alignment.shape != progress.shape or alignment.shape != groups.shape or not len(groups):
        raise ValueError('Utility/progress/group_ids must be equal nonempty vectors')
    if not np.isfinite(alignment).all() or not np.isfinite(progress).all() or (progress < 0.).any() or (groups < 0).any():
        raise ValueError('Invalid utility inputs')
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    alignment_z, alignment_fallback = _group_robust_z(alignment, inverse, counts)
    progress_z, progress_fallback = _group_robust_z(progress, inverse, counts)
    logits = np.clip(alignment_z + .25 * progress_z, -2., 2.)
    weights = torch.from_numpy(np.exp(logits).astype(np.float32))
    return weights, dict(normalization='group median / (1.4826 MAD); zero MAD -> centered RMS -> 1',
                         nonempty_groups=len(counts), alignment_zero_mad_groups=alignment_fallback,
                         progress_zero_mad_groups=progress_fallback, progress_coefficient=.25,
                         temperature=1., logit_clip=[-2., 2.], native_mix_applied=False)


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def score_dynamic_utility(model, optimizer, candidate_batches, reference_batches, candidate_count, *,
                          group_ids, step, previous_losses=None, reference_count=1024, provenance=None):
    """All ranks call together. Iterators yield (full batch, global candidate IDs).

    Each rank sees the same ordered batch iterator. Whole batch ordinal % world
    determines its scoring owner; forward batch geometry is never rank-split.
    Reference IDs can be candidate keys, but must cover the supplied TRAIN reference
    asset exactly; ownership is by reference batch ordinal. Inputs are prepared
    independently from the training window, which this function never consumes.
    Results are full CPU vectors on each rank, suitable for checkpointing. F invokes
    once with previous_losses=None and retains weights/losses; D invokes every 200
    optimizer updates and passes the last losses. R never calls the scorer.
    """
    if candidate_count <= 0 or reference_count <= 0:
        raise ValueError('Candidate and reference counts must be positive')
    if len(group_ids) != candidate_count:
        raise ValueError('group_ids must cover the complete candidate pool')
    distributed = dist.is_available() and dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    device = next(model.parameters()).device
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    collective_seconds = 0.
    def collective(value, broadcast=False):
        nonlocal collective_seconds
        if distributed:
            synchronize()
            started = time.perf_counter()
            if broadcast:
                dist.broadcast(value, src=0)
            else:
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
            synchronize()
            collective_seconds += time.perf_counter() - started
    synchronize()
    started, cpu_started = time.perf_counter(), time.process_time()
    with preserved_scoring_state(model), torch.inference_mode():
        preconditioner, adam_facts = adam_output_preconditioner(model, optimizer)
        sums = [torch.zeros_like(matrix) for matrix in preconditioner]
        count = torch.zeros((), device=device, dtype=torch.int64)
        reference_batches_owned = 0
        for ordinal, (batch, ids) in enumerate(reference_batches):
            if ordinal % world != rank:
                continue
            hidden, errors, _, _ = forward_factors(model, batch)
            if len(ids) != hidden.shape[0]:
                raise ValueError('Reference indices/batch count mismatch')
            for accumulator, gradient in zip(sums, reference_gradient_sums(hidden, errors)):
                accumulator.add_(gradient)
            count.add_(hidden.shape[0])
            reference_batches_owned += 1
        collective(count)
        if int(count) != reference_count:
            raise ValueError(f'Expected {reference_count} TRAIN reference examples, got {int(count)}')
        for gradient in sums:
            collective(gradient)
            gradient.div_(reference_count)
            collective(gradient, broadcast=True)
        reference_hashes = [_digest(gradient) for gradient in sums]
        weighted_reference = [gradient * diagonal for gradient, diagonal in zip(sums, preconditioner)]
        del preconditioner
        local = torch.zeros((5, candidate_count), dtype=torch.float32, device='cpu')
        batches_owned = 0
        for ordinal, (batch, ids) in enumerate(candidate_batches):
            if ordinal % world != rank:
                continue
            ids = torch.as_tensor(ids, dtype=torch.int64, device='cpu')
            if ids.ndim != 1 or not len(ids) or int(ids.min()) < 0 or int(ids.max()) >= candidate_count or ids.unique().numel() != len(ids):
                raise ValueError('Invalid candidate batch global indices')
            if bool((local[3, ids] != 0).any()):
                raise ValueError('Repeated candidate index in query pass')
            hidden, errors, losses, cost = forward_factors(model, batch)
            if len(ids) != hidden.shape[0]:
                raise ValueError('Candidate indices/batch count mismatch')
            raw = contract_alignment(hidden, errors, weighted_reference) / cost
            local[0, ids], local[1, ids], local[2, ids], local[3, ids] = raw.cpu(), losses.cpu(), cost.cpu(), 1.
            local[4,ids]=(batch[0].transformed_sequences['sequence_data']!=-1).sum(1).float().cpu()
            batches_owned += 1
        # Chunked small collectives avoid creating an N x parameter gradient matrix.
        for offset in range(0, candidate_count, 65536):
            piece = local[:, offset:offset + 65536].to(device)
            collective(piece)
            local[:, offset:offset + 65536] = piece.cpu()
        if not bool((local[3] == 1).all()) or not bool(torch.isfinite(local).all()) or not bool((local[2] > 0).all()):
            raise ValueError('Incomplete/overlapping/nonfinite full-pool scoring pass')
        alignment, losses, costs = [local[index].clone() for index in range(3)]
        if previous_losses is None:
            progress = torch.zeros_like(losses)
        else:
            previous = torch.as_tensor(previous_losses).detach().cpu().float()
            if previous.shape != losses.shape or not bool(torch.isfinite(previous).all()):
                raise ValueError('Previous CE vector must be finite and cover the same candidate pool')
            progress = (previous - losses).clamp_min(0.)
        weights, normalization = grouped_utility_weights(alignment, progress, group_ids)
        synchronize()
        wall = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started
        rank_costs = None
        if distributed:
            rank_costs = [None] * world
            dist.all_gather_object(rank_costs, dict(rank=rank, scoring_wall_seconds=wall,
                                  collective_wait_seconds=collective_seconds, cpu_seconds=cpu_seconds,
                                  query_batches=batches_owned, reference_batches=reference_batches_owned))
        else:
            rank_costs = [dict(rank=rank, scoring_wall_seconds=wall, collective_wait_seconds=0., cpu_seconds=cpu_seconds,
                              query_batches=batches_owned, reference_batches=reference_batches_owned)]
        synchronize()
        metadata = dict(format_version=1, step=int(step), candidate_count=candidate_count,
                        reference_count=reference_count, reference_split='TRAIN only',
                        method='training-reference Adam-diagonal output-head alignment / constant candidate slot cost',
                        output_heads='distinct parameter blocks; independent SID level output matrices',
                        gradient='per-example mean CE over valid SID targets; concatenated output-head weights',
                        approximation='first-order output-layer-local; not full-model influence or full Adam update; no momentum',
                        denominator='constant 1 per candidate slot: actual eager T5 fixed padded encoder120(+fixed user slot)/four SID targets(+fixed BOS), no unpadding',
                        nonpadding_input_sid_statistics=dict(mean=float(local[4].mean()),minimum=float(local[4].min()),maximum=float(local[4].max()),
                            scope='Descriptive exposure only; never divides alignment or sampler weights'),
                        adam=adam_facts, reference_gradient_sha256=reference_hashes,
                        loss_sha256=_digest(losses), weight_sha256=_digest(weights),
                        normalization=normalization, first_score_progress_zero=previous_losses is None,
                        world_size=world, rank_costs=rank_costs,
                        scoring_wall_seconds=time.perf_counter() - started,
                        cost_includes='input loading, reference/query forward, normalization, collective and peer waiting',
                        provenance=provenance or {}, rng_and_training_modes_restored=True)
    # Return ordinary tensors, so callers may checkpoint and update state outside
    # inference_mode without restrictions on in-place operations.
    weights, losses, alignment, progress, costs = [value.clone() for value in (weights, losses, alignment, progress, costs)]
    return dict(weights=weights, positive_weights=weights, losses=losses, alignment_raw=alignment,
                progress=progress, cost_proxy=costs, metadata=metadata)
