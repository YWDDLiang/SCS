"""Original-batch work partition and unaltered max-reference gradient arithmetic."""
import math
import torch


def work_plan(count, batch_size, reference_sample, seed=42, pilot_batches=None):
    """Sample references from the entire pool, and retain complete original batches."""
    if count<=1 or batch_size<=0 or not 2<=reference_sample<=count:
        raise ValueError('Require pool>1, positive batch size and 2<=reference_sample<=pool')
    total_batches=math.ceil(count/batch_size)
    if pilot_batches is None:query_batches=torch.arange(total_batches)
    else:
        if not 0<pilot_batches<=total_batches:raise ValueError('pilot-batches exceeds available original batches')
        query_batches=torch.randperm(total_batches,generator=torch.Generator().manual_seed(seed ^ 0x51A7))[:pilot_batches].sort().values
    reference=torch.randperm(count,generator=torch.Generator().manual_seed(seed))[:reference_sample].sort().values
    forward_batches=torch.unique(torch.cat([query_batches,reference//batch_size]),sorted=True)
    def rows(batches):
        if not len(batches):return torch.empty(0,dtype=torch.int64)
        return torch.cat([torch.arange(int(b)*batch_size,min((int(b)+1)*batch_size,count)) for b in batches])
    queries=rows(query_batches)
    ranks=[]
    for rank in range(4):
        forward=rows(forward_batches[forward_batches%4==rank]);query=rows(query_batches[query_batches%4==rank])
        local_ref=reference[(reference//batch_size)%4==rank]
        ranks.append(dict(rank=rank,forward_indices=forward,query_indices=query,reference_indices=local_ref))
    return dict(candidate_count=count,batch_size=batch_size,reference_indices=reference,query_indices=queries,
                query_batches=query_batches,forward_batches=forward_batches,ranks=ranks,
                experiment_seed=seed,reference_seed=seed,pilot_batch_seed=seed ^ 0x51A7,random_control_seed=seed ^ 0xC0FFEE)


def merge_factors(shards, reference):
    ids=torch.cat([row['indices'] for row in shards]);order=torch.argsort(ids,stable=True)
    if not torch.equal(ids[order],reference):raise RuntimeError('Reference shards do not form the exact shared random set')
    return dict(indices=reference,errors=torch.cat([row['errors'] for row in shards])[order],hidden=torch.cat([row['hidden'] for row in shards])[order])


def redundancy_queries(errors, hidden, query_indices, reference_errors, reference_hidden, reference_indices, device, chunk):
    """Same fp16 factors/fp32 inner-products/max/self-mask as original _redundancy.

    Only indexing differs: original local ref_positions is derived from global
    candidate IDs, allowing partitioned queries against one shared reference.
    No gradient averaging, norm, Adam normalization, or scoring rescaling.
    """
    count,levels,_=errors.shape
    reference_positions=torch.searchsorted(reference_indices,query_indices)
    inside=reference_positions<reference_indices.numel()
    inside &= reference_indices[reference_positions.clamp(max=reference_indices.numel()-1)]==query_indices
    ref_positions=torch.where(inside,reference_positions,torch.full_like(reference_positions,-1))
    ref_errors=reference_errors.to(device);ref_hidden=reference_hidden.to(device)
    worst=torch.full((count,),float('-inf'),dtype=torch.float32)
    for start in range(0,count,chunk):
        stop=min(start+chunk,count)
        similarity=torch.zeros(stop-start,ref_errors.shape[0],dtype=torch.float32,device=device)
        for level in range(levels):
            term_e=errors[start:stop,level].to(device).float() @ ref_errors[:,level].float().T
            term_h=hidden[start:stop,level].to(device).float() @ ref_hidden[:,level].float().T
            similarity += term_e*term_h
            del term_e,term_h
        position=ref_positions[start:stop];inside=position>=0
        if bool(inside.any()):
            rows=inside.nonzero(as_tuple=True)[0]
            similarity[rows,position[inside]]=float('-inf')
        worst[start:stop]=similarity.amax(dim=1).cpu()
        del similarity
    if not torch.isfinite(worst).all():raise RuntimeError('redundancy contains non-finite values')
    return worst


def merge_selection(shards, query_indices, budget, random_seed):
    ids=torch.cat([row['indices'] for row in shards]);scores=torch.cat([row['redundancy'] for row in shards])
    order=torch.argsort(ids,stable=True);ids=ids[order];scores=scores[order]
    if not torch.equal(ids,query_indices) or ids.unique().numel()!=len(ids):raise RuntimeError('Score shard union/order mismatch')
    if not 0<budget<len(ids):raise ValueError('Pruning budget must be between zero and query size')
    deletion=torch.argsort(scores,descending=True,stable=True)
    keep=torch.ones(len(ids),dtype=torch.bool);keep[deletion[:len(ids)-budget]]=False
    selected=ids[keep]
    random=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(random_seed))[:budget]].sort().values
    return dict(indices=ids,redundancy=scores,algorithm=selected,random=random)


def tensor_equivalence(left, right, rtol=1e-6, atol=1e-7):
    """Fixed-tolerance audit with exact flag and bounded mismatch coordinates."""
    result=dict(passed=False,exact=False,rtol=rtol,atol=atol,left_shape=list(left.shape),right_shape=list(right.shape),left_dtype=str(left.dtype),right_dtype=str(right.dtype))
    if left.shape!=right.shape or left.dtype!=right.dtype:return result
    result['exact']=torch.equal(left,right)
    if not left.is_floating_point():
        result['passed']=result['exact'];return result
    delta=(left.double()-right.double()).abs()
    finite=bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    mask=~torch.isclose(left,right,rtol=rtol,atol=atol)
    absolute=float(delta.max()) if delta.numel() else 0.
    relative=float((delta/right.double().abs().clamp_min(1e-30)).max()) if delta.numel() else 0.
    result.update(passed=finite and torch.allclose(left,right,rtol=rtol,atol=atol),finite=finite,
        max_absolute_error=absolute if math.isfinite(absolute) else None,
        max_relative_error=relative if math.isfinite(relative) else None,
        mismatched_elements=int(mask.sum()),first_mismatch_coordinates=mask.nonzero()[:20].tolist())
    return result


def original_formula_audit(original_errors,original_hidden,distributed_factors,reference,query_indices,distributed_scores,budget,random_seed,device,chunk):
    """Audit original _redundancy directly on the same complete-batch union.

    Scoring factors remain immutable; the audit never substitutes baseline scores
    or selections for distributed outputs when a comparison fails.
    """
    from scripts.select_nacs_redundancy_candidates import _redundancy
    ids=distributed_factors['indices']
    ref_positions=torch.searchsorted(ids,reference['indices'])
    query_positions=torch.searchsorted(ids,query_indices)
    if not torch.equal(ids[ref_positions],reference['indices']) or not torch.equal(ids[query_positions],query_indices):
        raise RuntimeError('Audit batch union is missing query/reference rows')
    checks=dict(errors=tensor_equivalence(distributed_factors['errors'],original_errors),hidden=tensor_equivalence(distributed_factors['hidden'],original_hidden),
        shared_reference_errors=tensor_equivalence(reference['errors'],original_errors[ref_positions]),
        shared_reference_hidden=tensor_equivalence(reference['hidden'],original_hidden[ref_positions]))
    baseline_all=_redundancy(original_errors,original_hidden,ref_positions,device,chunk)
    baseline_scores=baseline_all[query_positions]
    checks['scores']=tensor_equivalence(distributed_scores['redundancy'],baseline_scores)
    original_selection=merge_selection([dict(indices=query_indices,redundancy=baseline_scores)],query_indices,budget,random_seed)
    distributed_selection=merge_selection([distributed_scores],query_indices,budget,random_seed)
    checks['algorithm_selection']=tensor_equivalence(distributed_selection['algorithm'],original_selection['algorithm'])
    checks['random_control']=tensor_equivalence(distributed_selection['random'],original_selection['random'])
    return dict(passed=all(value['passed'] for value in checks.values()),checks=checks,
        original_forward_rows=len(ids),query_rows=len(query_indices),reference_rows=len(reference['indices']),
        rtol=1e-6,atol=1e-7,baseline_original_function='scripts.select_nacs_redundancy_candidates._redundancy',
        no_output_substitution=True),dict(indices=query_indices,redundancy=baseline_scores,algorithm=original_selection['algorithm'],random=original_selection['random'])


def original_chunk_score_shard(factor_shards,universe,query_indices,reference,rank,device,chunk):
    """Partition original full GEMM chunks, never compact each rank's query rows.

    Preserve the original _redundancy M dimension, row placement and final-tail
    length. CPU/mmap factor shards supply full frames; only real query outputs
    are retained. Each required full chunk is computed by one rank, not four.
    """
    scored_ids=[];scored_values=[];frame_sizes=[]
    levels=factor_shards[0]['errors'].shape[1];vocab=factor_shards[0]['errors'].shape[2];hidden_dim=factor_shards[0]['hidden'].shape[2]
    for start in range(rank*chunk,len(universe),4*chunk):
        ids=universe[start:min(start+chunk,len(universe))]
        positions=torch.searchsorted(query_indices,ids)
        is_query=positions<len(query_indices)
        is_query &= query_indices[positions.clamp(max=len(query_indices)-1)]==ids
        if not bool(is_query.any()):continue
        errors=torch.empty(len(ids),levels,vocab,dtype=torch.float16);hidden=torch.empty(len(ids),levels,hidden_dim,dtype=torch.float16)
        filled=torch.zeros(len(ids),dtype=torch.bool)
        for shard in factor_shards:
            local=torch.searchsorted(shard['indices'],ids)
            inside=local<len(shard['indices'])
            if len(shard['indices']):inside &= shard['indices'][local.clamp(max=len(shard['indices'])-1)]==ids
            if bool(inside.any()):
                errors[inside]=shard['errors'][local[inside]];hidden[inside]=shard['hidden'][local[inside]];filled |=inside
        if not bool(filled.all()):raise RuntimeError('Original score frame missing factor rows')
        scores=redundancy_queries(errors,hidden,ids,reference['errors'],reference['hidden'],reference['indices'],device,chunk)
        scored_ids.append(ids[is_query]);scored_values.append(scores[is_query]);frame_sizes.append(len(ids))
    return dict(indices=torch.cat(scored_ids) if scored_ids else torch.empty(0,dtype=torch.int64),
        redundancy=torch.cat(scored_values) if scored_values else torch.empty(0,dtype=torch.float32),frame_sizes=frame_sizes,
        query_chunk_policy='full_original_universe_chunks_mod4_v2')
