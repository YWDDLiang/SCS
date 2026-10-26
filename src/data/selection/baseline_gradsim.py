"""Frozen GradSim + Diverse-Weighted on the complete literal occurrence pool."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist

from src.data.selection.baseline_gradient_features import query_gradient_features, scoring_state
from src.data.selection.baseline_capacity_sampling import weighted_batch, choose_literal_ids


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(p)


def unpack(bank):
    """Four projection activation/backprop factors, stored in fp32."""
    pos, pairs = 0, []
    for width_in, width_out in [(128, 384)] * 3 + [(384, 128)]:
        a = bank[:, pos:pos+5*width_in].reshape(-1, 5, width_in)
        pos += 5*width_in
        b = bank[:, pos:pos+5*width_out].reshape(-1, 5, width_out)
        pos += 5*width_out
        pairs.append((a, b))
    if pos != bank.shape[1]:
        raise RuntimeError("Gradient factor layout mismatch")
    return pairs


def normalized_sum(factors, norms, local_keys, multiplicities, denominator):
    result = [torch.zeros((b.shape[2], a.shape[2]), dtype=torch.float64, device=a.device)
              for a, b in factors]
    if len(local_keys):
        keys = torch.as_tensor(local_keys, dtype=torch.long, device=norms.device)
        weights = torch.as_tensor(multiplicities, dtype=torch.float64, device=norms.device)
        selected_norms = norms[keys].double()
        weights = torch.where(selected_norms > 0, weights/selected_norms.clamp_min(1e-300),
                              torch.zeros_like(weights)) / denominator
        for i, (a, b) in enumerate(factors):
            result[i] = torch.einsum("rto,rti,r->oi", b[keys].double(),
                                     a[keys].double(), weights)
    # einsum can return a transposed layout when exactly one row is selected.
    # Every rank must reduce the same contiguous logical coordinate layout.
    return [value.contiguous() for value in result]


def collect_factors(module, destination, triples, lo, hi):
    model = module.trainer.lightning_module
    rank = int(module.trainer.global_rank)
    bank_path = destination / f"factors_rank{rank}.npy"
    norm_path = destination / f"norms_rank{rank}.npy"
    if bank_path.exists() or norm_path.exists():
        raise RuntimeError("Preserve partial scoring attempt; do not silently overwrite")
    bank = np.lib.format.open_memmap(bank_path, mode="w+", dtype=np.float32, shape=(hi-lo, 10240))
    norms = np.lib.format.open_memmap(norm_path, mode="w+", dtype=np.float64, shape=(hi-lo,))
    for group in range(lo, hi):
        batch = module._fast_collate(torch.tensor(np.array(triples[group:group+1]), dtype=torch.int32))
        feature = query_gradient_features(model, batch, need_factors=True, need_full_norm=False)
        arrays = []
        for a, b in feature["factors"]:
            if a.shape[0] != 5:
                raise RuntimeError("Unexpected decoder length; no projection approximation allowed")
            arrays += [a.cpu().numpy().reshape(-1), b.cpu().numpy().reshape(-1)]
        bank[group-lo] = np.concatenate(arrays)
        norms[group-lo] = float(feature["attention_norm"])
        if (group-lo+1) % 512 == 0:
            atomic(destination / f"features_progress_rank{rank}.json",
                   dict(done=group-lo+1, total=hi-lo, phase="actual_step200_gradient_factors"))
    bank.flush()
    norms.flush()
    if not np.isfinite(bank).all() or not np.isfinite(norms).all():
        raise RuntimeError("Incomplete/nonfinite attention factors")
    return bank, norms


def select_capacities(bank, norms, counts, reference_keys, lo, hi, rank, world, destination):
    device = norms.device
    total_groups = len(counts)
    width = (total_groups+world-1)//world
    factors = bank if isinstance(bank, (tuple, list)) else [
        (a.contiguous(), b.contiguous()) for a, b in unpack(bank)]
    ref_keys, ref_counts = np.unique(reference_keys, return_counts=True)
    mask = (ref_keys >= lo) & (ref_keys < hi)
    direction = normalized_sum(factors, norms, ref_keys[mask]-lo, ref_counts[mask],
                               len(reference_keys))
    for d in direction:
        dist.all_reduce(d)
    capacities = np.asarray(counts, dtype=np.int64).copy() if rank == 0 else None
    selected = np.zeros(total_groups, dtype=np.int64) if rank == 0 else None
    wanted = 4*int(np.sum(counts))//5
    order = (np.lib.format.open_memmap(destination / "selected_key_order.npy", mode="w+",
                                      dtype=np.uint32, shape=(wanted,))
             if rank == 0 else None)
    rng = np.random.default_rng(42 ^ 0xD1E57)
    scores_local = torch.empty(width, dtype=torch.float64, device=device)
    scores_all = torch.empty(width*world, dtype=torch.float64, device=device)
    chosen_tensor = torch.empty(128, dtype=torch.int64, device=device)
    started = time.perf_counter()
    for done in range(0, wanted, 128):
        size = min(128, wanted-done)
        scores_local.fill_(0)
        values = torch.zeros(hi-lo, dtype=torch.float64, device=device)
        for (a, b), d in zip(factors, direction):
            # All factors remain fp32, as specified. Accumulated reference
            # direction and final normalization use fp64 for numerical range.
            projected = (a.reshape(-1, a.shape[-1]) @ d.T.float()).reshape_as(b)
            values += (projected * b).sum((1, 2)).double()
        values = torch.where(norms > 0, values/norms.clamp_min(1e-300), torch.zeros_like(values))
        scores_local[:hi-lo] = values
        dist.all_gather_into_tensor(scores_all, scores_local)
        if rank == 0:
            score = scores_all[:total_groups].cpu().numpy()
            active = capacities > 0
            weight = np.zeros(total_groups, dtype=np.float64)
            weight[active] = score[active] - float(score[active].min()) + 1e-8
            if total_groups <= 32 and done == 0:
                atomic(destination / "small_fixture_sampling_input.json",
                       dict(scores=score.tolist(), weights=weight.tolist(),
                            capacities=capacities.tolist(), rng=rng.bit_generator.state))
            choice = weighted_batch(capacities, weight, size, rng)
            chosen_keys, taken = np.unique(choice, return_counts=True)
            capacities[chosen_keys] -= taken
            selected[chosen_keys] += taken
            order[done:done+size] = choice
            if np.any(capacities[chosen_keys] < 0):
                raise RuntimeError("Weighted selection exceeded occurrence capacity")
            chosen_tensor[:size] = torch.as_tensor(choice, device=device)
        dist.broadcast(chosen_tensor, src=0)
        if done+size < wanted:
            choice = chosen_tensor[:size].cpu().numpy()
            chosen_keys, taken = np.unique(choice, return_counts=True)
            owned = (chosen_keys >= lo) & (chosen_keys < hi)
            updates = normalized_sum(factors, norms, chosen_keys[owned]-lo,
                                     taken[owned], size)
            for d, update in zip(direction, updates):
                dist.all_reduce(update)
                d -= update
        if rank == 0 and ((done//128) % 1000 == 0 or done+size == wanted):
            atomic(destination / "selection_progress.json",
                   dict(phase="capacity_preserving_diverse_weighted",
                        selected_occurrences=done+size, target_occurrences=wanted,
                        elapsed_seconds=time.perf_counter()-started,
                        remaining_groups=int(np.count_nonzero(capacities))))
    if rank == 0:
        if int(selected.sum()) != wanted or np.any(selected > counts):
            raise RuntimeError("Final selected multiset has wrong cardinality")
        order.flush()
        np.save(destination / "selected_key_counts.npy", selected)
    return selected


def ensure_selection(module):
    trainer = module.trainer
    rank, world = int(trainer.global_rank), int(trainer.world_size)
    if trainer.global_step != 200 or module.consumed_batches != 25600 or world != 4:
        raise RuntimeError("Approved continuous step200/four-rank selection required")
    out = module.selection_output_dir
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    trainer.strategy.barrier("baseline_selection_directory")
    if (out / "manifest.json").exists():
        # Caller verifies the finished data artifact. Resume from an older
        # model checkpoint is handled explicitly, not inferred from this file.
        raise RuntimeError("Existing finished selection requires explicit continuation binding")
    start, cpu = time.perf_counter(), time.process_time()
    index = Path(os.environ.get("REPLAY_SELECTION_CACHE", "/tmp/replay_selection_cache")) / module.dataset / "gradient_index"
    info = json.loads((index / "manifest.json").read_text())
    if (info.get("ready") is not True
            or info["source_asset_identity"] != module.asset_manifest["immutable_identity"]
            or info["source_manifest_sha256"] != sha(module.assets_path / "manifest.json")):
        raise RuntimeError("Effective-input index belongs to a different literal pool")
    for name, facts in info["files"].items():
        if sha(index / name) != facts["sha256"]:
            raise RuntimeError("Computational index changed: " + name)
    triples = np.load(index / "unique_effective_occurrences.npy", mmap_mode="r")
    counts = np.load(index / "key_counts.npy", mmap_mode="r")
    mapping = np.load(index / "occurrence_to_key.npy", mmap_mode="r")
    if int(counts.sum()) != module.occurrence_count:
        raise RuntimeError("Occurrence capacity mismatch")
    required_storage = len(counts)*10240*4 + module.occurrence_count*12 + 10*1024**3
    if shutil.disk_usage(out).free < required_storage:
        raise RuntimeError("Insufficient output space for complete factors and selection evidence")
    width = (len(counts)+world-1)//world
    lo, hi = rank*width, min((rank+1)*width, len(counts))
    ref_rng = np.random.default_rng(42 ^ 0x51A7)
    reference_ids = ref_rng.choice(module.occurrence_count, size=1024, replace=False)
    reference_keys = np.asarray(mapping[reference_ids], dtype=np.int64)
    model = trainer.lightning_module
    device = next(model.parameters()).device
    from src.data.loading.datamodules.replay_selection_datamodule import _state_hash
    from src.callbacks.ddp_schedule_tracking import capture_rng
    def state_observation():
        return dict(model_sha256=_state_hash(model.state_dict()),
                    Adam_sha256=_state_hash(trainer.optimizers[0].state_dict()),
                    RNG_sha256=_state_hash(capture_rng()),
                    cursor=module.consumed_batches)
    before = state_observation()
    with scoring_state(model, trainer.strategy.model):
        bank_array, norm_array = collect_factors(module, out, triples, lo, hi)
        bank = torch.tensor(np.array(bank_array), dtype=torch.float32, device=device)
        factors = [(a.contiguous(), b.contiguous()) for a, b in unpack(bank)]
        del bank
        norms = torch.tensor(np.array(norm_array), dtype=torch.float64, device=device)
        trainer.strategy.barrier("all_rank_gradient_factors_ready")
        selected = select_capacities(factors, norms, counts, reference_keys,
                                     lo, hi, rank, world, out)
        del factors, norms
        if rank == 0:
            literal_rng = np.random.default_rng(42 ^ 0x1D3A71)
            membership = choose_literal_ids(counts, selected, mapping, literal_rng)
            positions = np.flatnonzero(membership[np.asarray(module.permutation)])
            np.save(out / "selected_positions.npy", positions)
            np.save(out / "reference_occurrence_ids.npy", reference_ids)
        trainer.strategy.barrier("literal_membership_projection_ready")
    after = state_observation()
    torch.cuda.synchronize(device)
    factor_hash = sha(out / f"factors_rank{rank}.npy")
    norm_hash = sha(out / f"norms_rank{rank}.npy")
    cost = dict(rank=rank, score_device_seconds=time.perf_counter()-start,
                CPU_seconds=time.process_time()-cpu,
                state_before=before, state_after=after,
                state_comparison_informational_only=True,
                factors_sha256=factor_hash, norms_sha256=norm_hash)
    costs = [None]*world
    dist.all_gather_object(costs, cost)
    if rank == 0:
        publication_cpu, publication_wall = time.process_time(), time.perf_counter()
        source = module.asset_manifest
        positions_path = out / "selected_positions.npy"
        manifest = dict(
            format_version=1, ready=True, method="GradSim_Diverse_Weighted",
            baseline_protocol_sha256=module.baseline_protocol_sha256,
            total_occurrences=module.occurrence_count, selected_count=4*module.occurrence_count//5,
            source_stream_id=source["source_stream_id"],
            replay_shuffle_asset_identity=source["immutable_identity"],
            source_manifest_sha256=sha(module.assets_path / "manifest.json"),
            permutation_sha256=source["files"]["permutation.npy"]["sha256"],
            files={"selected_positions.npy": dict(path=str(positions_path),
                                                  sha256=sha(positions_path))},
            scoring_step=200, reference_source="TRAIN occurrence population",
            reference_count=1024, sample_batch=128,
            capacities_preserved=True, training_deduplicated=False,
            full_gradient_parameter_mapping="last decoder self-attention q/k/v/o",
            parameter_factors="fp32 exact linear gradients; no random projection",
            costs=costs, score_time_also_in_loader_wait_and_process=True,
            unique_effective_input_groups=len(counts),
            index_manifest_sha256=sha(index / "manifest.json"),
            selection_evidence_sha256={name: sha(out / name) for name in
                                       ["selected_key_counts.npy", "selected_key_order.npy",
                                        "reference_occurrence_ids.npy"]})
        manifest["publication_cost"] = dict(
            CPU_seconds=time.process_time()-publication_cpu,
            wall_seconds=time.perf_counter()-publication_wall,
            overlaps_loader_wait_process_wrapper=True)
        atomic(out / "manifest.json", manifest)
    trainer.strategy.barrier("baseline_selection_manifest_ready")
    return out
