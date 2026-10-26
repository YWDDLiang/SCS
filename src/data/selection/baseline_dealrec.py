"""Approved DEALRec migration on the complete shared offline occurrence pool."""
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist

from scripts.run_baseline_sasrec import OfflineQueries, load_author_model
from src.data.selection.baseline_gradsim import atomic, sha
from src.data.selection.baseline_gradient_features import scoring_state, query_gradient_features
from src.data.selection.baseline_dealrec_core import (
    PARAMETER, QueryLosses, influence_parameter, sample_negatives,
    original_microbatch_weights, inverse_hvp, directional_influence, coverage_selection)


def interval(n, rank, world):
    return n * rank // world, n * (rank + 1) // world


def tensors(queries, rows, rng, device):
    x, positive, users = queries.from_rows(rows)
    negative = sample_negatives(queries, users, rng)
    batch = tuple(torch.tensor(np.array(v), device=device, dtype=torch.long)
                  for v in (x, positive, negative))
    return batch, negative


def load_accepted_proxy(root, dataset, queries, device):
    run = root / "runs" / f"{dataset}_sasrec_for_dealrec_s42"
    acceptance = json.loads((run / "completion_accepted.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    checkpoint = run / "checkpoints/best.ckpt"
    if (acceptance.get("status") != "passed" or not acceptance.get("backup_verified")
            or acceptance["summary_sha256"] != sha(run / "summary.json")
            or acceptance["roles"]["best"]["sha256"] != sha(checkpoint)
            or summary["source_id"] != queries.source_id
            or summary["stop_reason"] != "early_stopping"):
        raise RuntimeError("DEALRec needs the audited, backed-up same-source proxy")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    backbone = load_author_model(root, queries.item_count, device.type == "cuda")
    backbone.load_state_dict(saved["model"], strict=True)
    losses = QueryLosses(backbone.to(device)).eval()
    for name, parameter in losses.named_parameters():
        parameter.requires_grad_(name == PARAMETER)
    evidence = dict(path=str(checkpoint), sha256=sha(checkpoint),
                    best_step=int(saved["global_step"]),
                    completion_accepted_sha256=sha(run / "completion_accepted.json"),
                    summary_sha256=sha(run / "summary.json"))
    return losses, evidence


def surrogate_scores(queries, losses, out, rank, world, depth=5000, batch_size=512):
    """One full v pass, author HVP recursion, one full literal influence pass.

    P positions are partitioned for computation. v retains original K weights;
    HVP samples uniform literal IDs with replacement. Every loss call draws a
    TRAIN-only negative, exactly as in the auxiliary current-query objective.
    """
    parameter = influence_parameter(losses)
    device = parameter.device
    n = len(queries.occ)
    lo, hi = interval(n, rank, world)
    generator_v = np.random.default_rng(42 ^ 0xDEA100 ^ rank)
    generator_score = np.random.default_rng(42 ^ 0xDEA200 ^ rank)
    v_neg_path = out / f"v_negative_rank{rank}.npy"
    score_neg_path = out / f"influence_negative_rank{rank}.npy"
    score_path = out / f"influence_rank{rank}.npy"
    for path in (v_neg_path, score_neg_path, score_path):
        if path.exists():
            raise RuntimeError("Preserve partial DEALRec scores; no implicit overwrite/resume")
    v_negatives = np.lib.format.open_memmap(
        v_neg_path, mode="w+", dtype=np.int32, shape=(hi - lo,))
    score_negatives = np.lib.format.open_memmap(
        score_neg_path, mode="w+", dtype=np.int32, shape=(hi - lo,))
    scores = np.lib.format.open_memmap(
        score_path, mode="w+", dtype=np.float64, shape=(hi - lo,))
    vector = torch.zeros_like(parameter, dtype=torch.float64)
    phases = {}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sync()
    began = time.perf_counter()
    for start in range(lo, hi, batch_size):
        end = min(start + batch_size, hi)
        positions = np.arange(start, end, dtype=np.int64)
        rows = queries.occ[queries.permutation[start:end]]
        batch, negative = tensors(queries, rows, generator_v, device)
        v_negatives[start - lo:end - lo] = negative
        weights = torch.as_tensor(
            original_microbatch_weights(queries.k_offsets, positions),
            device=device, dtype=torch.float64)
        loss = (losses(*batch).double() * weights).sum()
        gradient, = torch.autograd.grad(loss, parameter)
        vector += gradient.detach().double()
        if (start - lo) // batch_size % 100 == 0 or end == hi:
            atomic(out / f"influence_progress_rank{rank}.json",
                   dict(phase="constant_vector_original_P_K", done=end - lo,
                        total=hi - lo, elapsed_seconds=time.perf_counter() - began))
    v_negatives.flush()
    sync()
    phases["constant_vector_compute_wall_seconds"] = time.perf_counter() - began
    dist.all_reduce(vector)
    # Author computation uses model dtype for the inverse-Hessian recursion.
    vector = vector.to(parameter.dtype)
    if not torch.isfinite(vector).all():
        raise RuntimeError("Non-finite DEALRec constant vector")
    estimate = torch.zeros_like(parameter)
    began = time.perf_counter()
    if rank == 0:
        generator_hvp = np.random.default_rng(42 ^ 0xDEA300)
        ids = generator_hvp.integers(0, n, size=depth, dtype=np.int64)
        negatives = np.empty(depth, np.int32)

        def loss_at(step):
            batch, negative = tensors(
                queries, queries.occ[ids[step:step + 1]], generator_hvp, device)
            negatives[step] = negative[0]
            return losses(*batch).mean()

        def progress(step, value):
            if step % 100 == 0 or step == depth:
                atomic(out / "hvp_progress.json",
                       dict(done=step, total=depth, norm=float(value.double().norm()),
                            phase="author_inverse_HVP", elapsed_seconds=time.perf_counter() - began))

        estimate = inverse_hvp(vector, parameter, loss_at, depth=depth, progress=progress)
        np.save(out / "hvp_occurrence_ids.npy", ids)
        np.save(out / "hvp_negative_items.npy", negatives)
        np.save(out / "constant_vector.npy", vector.detach().cpu().numpy())
        np.save(out / "inverse_hvp.npy", estimate.detach().cpu().numpy())
        sync()
    phases["HVP_compute_wall_seconds"] = time.perf_counter() - began if rank == 0 else 0.
    wait_begin = time.perf_counter()
    dist.broadcast(estimate, src=0)
    sync()
    phases["HVP_broadcast_or_wait_wall_seconds"] = time.perf_counter() - wait_begin
    began = time.perf_counter()
    for start in range(lo, hi, batch_size):
        end = min(start + batch_size, hi)
        batch, negative = tensors(queries, queries.occ[queries.permutation[start:end]],
                                  generator_score, device)
        scores[start - lo:end - lo] = directional_influence(
            losses, batch, estimate).double().cpu().numpy()
        score_negatives[start - lo:end - lo] = negative
        if (start - lo) // batch_size % 100 == 0 or end == hi:
            atomic(out / f"influence_progress_rank{rank}.json",
                   dict(phase="literal_influence", done=end - lo, total=hi - lo,
                        elapsed_seconds=time.perf_counter() - began))
    scores.flush()
    score_negatives.flush()
    sync()
    phases["influence_compute_wall_seconds"] = time.perf_counter() - began
    if not np.isfinite(scores).all():
        raise RuntimeError("Incomplete or non-finite literal influence")
    return phases


def ensure_selection(module):
    trainer = module.trainer
    rank, world = int(trainer.global_rank), int(trainer.world_size)
    if trainer.global_step != 200 or module.consumed_batches != 25600 or world != 4:
        raise RuntimeError("DEALRec requires the approved continuous update200/four ranks")
    out, root = module.selection_output_dir, module.baseline_root
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    trainer.strategy.barrier("DEALRec_output_directory")
    if (out / "manifest.json").exists():
        raise RuntimeError("Finished selection needs explicit continuation binding")
    protocol = json.loads((root / "frozen_protocol.json").read_text())
    fixed = protocol["fixed_selector_parameters"]["DEALRec"]
    expected = dict(HVP_depth=5000, HVP_repetitions=1, damp=.01, scale=25.,
                    strata=50, hard_prune=.1)
    if any(fixed[k] != v for k, v in expected.items()) or fixed["lambda"] != .3:
        raise RuntimeError("DEALRec frozen selector parameters differ")
    started, cpu = time.perf_counter(), time.process_time()
    index = Path(os.environ.get("REPLAY_SELECTION_CACHE", "/tmp/replay_selection_cache")) / module.dataset / "gradient_index"
    info = json.loads((index / "manifest.json").read_text())
    if (info.get("ready") is not True
            or info["source_asset_identity"] != module.asset_manifest["immutable_identity"]
            or info["source_manifest_sha256"] != sha(module.assets_path / "manifest.json")):
        raise RuntimeError("Effort index belongs to another literal pool")
    for name, facts in info["files"].items():
        if sha(index / name) != facts["sha256"]:
            raise RuntimeError("Effort index changed: " + name)
    triples = np.load(index / "unique_effective_occurrences.npy", mmap_mode="r")
    counts = np.load(index / "key_counts.npy", mmap_mode="r")
    mapping = np.load(index / "occurrence_to_key.npy", mmap_mode="r")
    n = module.occurrence_count
    if counts.sum() != n or len(mapping) != n:
        raise RuntimeError("Full literal population mismatch")
    if shutil.disk_usage(out).free < n * 32 + 5 * 1024**3:
        raise RuntimeError("Insufficient storage for complete influence/negative/selection evidence")
    model = trainer.lightning_module
    device = next(model.parameters()).device
    from src.data.loading.datamodules.replay_selection_datamodule import _state_hash
    from src.callbacks.ddp_schedule_tracking import capture_rng

    def observe():
        return dict(model_sha256=_state_hash(model.state_dict()),
                    Adam_sha256=_state_hash(trainer.optimizers[0].state_dict()),
                    RNG_sha256=_state_hash(capture_rng()), cursor=module.consumed_batches)

    before = observe()
    with scoring_state(model, trainer.strategy.model):
        queries = OfflineQueries(root, module.dataset)
        if queries.source_id != module.asset_manifest["immutable_identity"]:
            raise RuntimeError("Proxy query source differs from main training")
        proxy, proxy_evidence = load_accepted_proxy(root, module.dataset, queries, device)
        phases = surrogate_scores(queries, proxy, out, rank, world)
        del proxy, queries
        lo, hi = interval(len(counts), rank, world)
        effort_path = out / f"effort_rank{rank}.npy"
        if effort_path.exists():
            raise RuntimeError("Preserve existing effort scoring attempt")
        effort = np.lib.format.open_memmap(
            effort_path, mode="w+", dtype=np.float64, shape=(hi - lo,))
        began = time.perf_counter()
        for key in range(lo, hi):
            batch = module._fast_collate(
                torch.tensor(np.array(triples[key:key + 1]), dtype=torch.int32))
            feature = query_gradient_features(model, batch, need_factors=False,
                                              need_full_norm=True)
            effort[key - lo] = float(feature["full_norm"])
            if (key - lo + 1) % 512 == 0 or key + 1 == hi:
                atomic(out / f"effort_progress_rank{rank}.json",
                       dict(phase="actual_TIGER200_full_trainable_norm",
                            done=key - lo + 1, total=hi - lo,
                            elapsed_seconds=time.perf_counter() - began))
        effort.flush()
        torch.cuda.synchronize(device)
        phases["TIGER_effort_compute_wall_seconds"] = time.perf_counter() - began
        if not np.isfinite(effort).all():
            raise RuntimeError("Incomplete or non-finite TIGER effort")
        trainer.strategy.barrier("DEALRec_all_scores_ready")
        if rank == 0:
            publication_cpu, publication_wall = time.process_time(), time.perf_counter()
            all_effort = np.concatenate([
                np.load(out / f"effort_rank{r}.npy", mmap_mode="r") for r in range(world)])
            influence = np.concatenate([
                np.load(out / f"influence_rank{r}.npy", mmap_mode="r") for r in range(world)])
            literal_effort = np.empty(n, dtype=np.float64)
            for start in range(0, n, 1_000_000):
                end = min(start + 1_000_000, n)
                literal_effort[start:end] = all_effort[mapping[module.permutation[start:end]]]
            positions, allocation = coverage_selection(
                influence, literal_effort, 4 * n // 5, np.random.default_rng(42 ^ 0xDEA400))
            np.save(out / "selected_positions.npy", positions)
            atomic(out / "coverage_allocation.json", allocation)
            publication = dict(CPU_seconds=time.process_time() - publication_cpu,
                               wall_seconds=time.perf_counter() - publication_wall,
                               overlaps_rank_allocated_wait_and_process=True)
        trainer.strategy.barrier("DEALRec_literal_projection_ready")
    after = observe()
    rank_files = [f"{name}_rank{rank}.npy"
                  for name in ("influence", "influence_negative", "v_negative", "effort")]
    cost = dict(rank=rank, scoring_allocated_rank_wall_seconds=time.perf_counter() - started,
                CPU_seconds=time.process_time() - cpu, phases=phases,
                state_before=before, state_after=after, informational_state_comparison_only=True,
                artifact_sha256={name: sha(out / name) for name in rank_files})
    costs = [None] * world
    dist.all_gather_object(costs, cost)
    if rank == 0:
        source = module.asset_manifest
        manifest = dict(
            format_version=1, ready=True, method="DEALRec",
            baseline_protocol_sha256=module.baseline_protocol_sha256,
            total_occurrences=n, selected_count=4 * n // 5,
            source_stream_id=source["source_stream_id"],
            replay_shuffle_asset_identity=source["immutable_identity"],
            source_manifest_sha256=sha(module.assets_path / "manifest.json"),
            permutation_sha256=source["files"]["permutation.npy"]["sha256"],
            files={"selected_positions.npy": dict(path=str(out / "selected_positions.npy"),
                                                   sha256=sha(out / "selected_positions.npy"))},
            scoring_step=200, influence_parameter=PARAMETER, surrogate=proxy_evidence,
            v_weighting="mean original P/K microbatch means over one full literal pool pass",
            HVP_sampling="5000 uniform literal TRAIN IDs with replacement",
            negative_sampling="fresh per literal query per pass; excludes user's TRAIN items",
            negative_rng_seeds=dict(v="42 xor 0xDEA100 xor rank",
                                    influence="42 xor 0xDEA200 xor rank", HVP=42 ^ 0xDEA300),
            HVP_depth=5000, HVP_repetitions=1, damp=.01, scale=25.,
            final_inverse_HVP_divide_by_scale=False, full_trainable_TIGER_effort=True,
            effort_definition="norm of mean four target-token CE gradient",
            proxy_influence_score_copied_across_repeats=False, training_deduplicated=False,
            unique_effective_input_groups=len(counts), costs=costs, publication_cost=publication,
            cost_accounting="phase/allocated scoring/loader wait/process/wrapper overlap; do not add",
            allocation=allocation, index_manifest_sha256=sha(index / "manifest.json"),
            selection_evidence_sha256={name: sha(out / name) for name in
                                       ("coverage_allocation.json", "hvp_occurrence_ids.npy",
                                        "hvp_negative_items.npy", "constant_vector.npy",
                                        "inverse_hvp.npy")})
        atomic(out / "manifest.json", manifest)
    trainer.strategy.barrier("DEALRec_manifest_published")
    return out
