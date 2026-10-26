"""GORACS at the continuously trained TIGER step200, after full-scale admission."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from scripts.baseline_goracs_preparation import read, sha, atomic, alive
from src.data.selection.baseline_gradient_features import scoring_state, query_gradient_features


def verify_preparation_admission(root, dataset):
    """No small fixture or zero-gradient transport probe alone can open this gate."""
    prep = root / "preparation" / dataset / "goracs_gpu_prepare"
    path = root / "preparation" / dataset / "goracs_full_scale_admission.json"
    policy_path = root / "preparation" / "goracs_precheck_policy.json"
    if policy_path.exists():
        policy = read(policy_path)
        if (policy.get("status") == "independent_prechecks_cancelled_by_user"
                and policy.get("authorized") is True and dataset in policy.get("datasets", [])
                and policy.get("production_OT_ITRA_unchanged") is True):
            if (prep / "daemon.json").exists() and alive(read(prep / "daemon.json")):
                raise RuntimeError("GPU preparation must exit before main training")
            prepared = read(prep / "manifest.json")
            # Input identities, full dimensions, and actual selection validity
            # remain checked below; this is not a feasibility certificate.
            return prep, prepared, policy_path
    admission = read(path)
    if (admission.get("status") != "passed" or admission.get("full_pool_semantics") is not True
            or admission.get("full_pool_refinement_resource_review_passed") is not True
            or admission["preparation_manifest_sha256"] != sha(prep / "manifest.json")):
        raise RuntimeError("Complete real-scale GORACS feasibility not admitted")
    if (prep / "daemon.json").exists() and alive(read(prep / "daemon.json")):
        raise RuntimeError("GPU preparation must exit before main training")
    probe_path = root / "preparation" / dataset / "goracs_transport_probe/manifest.json"
    probe = read(probe_path)
    if (probe["status"] != "certified_full_scale_fixed_subset_transport"
            or admission["transport_probe_manifest_sha256"] != sha(probe_path)):
        raise RuntimeError("Complete real-scale exact transport certificate missing")
    resource_path = root / "preparation" / dataset / "goracs_ITRA_resource_probe/manifest.json"
    resources = read(resource_path)
    if (resources["status"] != "full_scale_distance_only_ITRA_resource_probe_completed"
            or admission["ITRA_resource_probe_manifest_sha256"] != sha(resource_path)
            or resources["preparation_manifest_sha256"] != sha(prep / "manifest.json")
            or resources["original_occurrences"] != probe["original_occurrences"]):
        raise RuntimeError("Complete-scale ITRA resource evidence missing")
    prepared = read(prep / "manifest.json")
    return prep, prepared, path


def ensure_selection(module):
    trainer, root = module.trainer, module.baseline_root
    rank, world = int(trainer.global_rank), int(trainer.world_size)
    if trainer.global_step != 200 or module.consumed_batches != 25600 or world != 4:
        raise RuntimeError("Original continuous step200/four-rank protocol required")
    protocol = read(root / "frozen_protocol.json")
    fixed = protocol["fixed_selector_parameters"]["GORACS"]
    if (fixed["reference_occurrences"] != 5000 or fixed["lambda"] != .1
            or fixed["max_improving_exchanges"] != 100
            or fixed["exchange_candidates_each_side"] != 30
            or not fixed["complete_ITRA_core_required"]):
        raise RuntimeError("GORACS frozen scientific parameters differ")
    prep, prepared, admission_path = verify_preparation_admission(root, module.dataset)
    out = module.selection_output_dir
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    trainer.strategy.barrier("GORACS_selection_directory")
    if (out / "manifest.json").exists():
        raise RuntimeError("Finished selection requires explicit continuation binding")
    started, cpu = time.perf_counter(), time.process_time()
    index = Path(os.environ.get("REPLAY_SELECTION_CACHE", "/tmp/replay_selection_cache")) / module.dataset / "gradient_index"
    info = read(index / "manifest.json")
    if (info.get("ready") is not True
            or info["source_asset_identity"] != module.asset_manifest["immutable_identity"]
            or sha(index / "manifest.json") != prepared["input_binding"]["source_index_manifest_sha256"]
            or info["source_manifest_sha256"] != sha(module.assets_path / "manifest.json")):
        raise RuntimeError("Full text/gradient input identities differ")
    for name in ("unique_effective_occurrences.npy", "key_counts.npy"):
        if sha(index / name) != info["files"][name]["sha256"]:
            raise RuntimeError("Complete gradient input index changed")
    triples = np.load(index / "unique_effective_occurrences.npy", mmap_mode="r")
    counts = np.load(index / "key_counts.npy", mmap_mode="r")
    n, groups = module.occurrence_count, len(counts)
    if int(counts.sum()) != n or prepared["distance"]["shape"] != [groups, 5000]:
        raise RuntimeError("Full original population/reference dimensions changed")
    if shutil.disk_usage(out).free < groups * 5000 * 4 + n * 17 + 5 * 1024**3:
        raise RuntimeError("Insufficient disk for full costs and literal selection evidence")
    model, device = trainer.lightning_module, next(trainer.lightning_module.parameters()).device
    from src.data.loading.datamodules.replay_selection_datamodule import _state_hash
    from src.callbacks.ddp_schedule_tracking import capture_rng

    def observe():
        return dict(model_sha256=_state_hash(model.state_dict()),
                    Adam_sha256=_state_hash(trainer.optimizers[0].state_dict()),
                    RNG_sha256=_state_hash(capture_rng()), cursor=module.consumed_batches)

    before = observe()
    with scoring_state(model, trainer.strategy.model):
        lo, hi = groups * rank // world, groups * (rank + 1) // world
        norm_path = out / f"norms_rank{rank}.npy"
        if norm_path.exists():
            raise RuntimeError("Existing norm pass must be preserved, not overwritten")
        norms = np.lib.format.open_memmap(norm_path, mode="w+", dtype=np.float64, shape=(hi - lo,))
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        for group in range(lo, hi):
            batch = module._fast_collate(torch.tensor(np.array(triples[group:group + 1]), dtype=torch.int32))
            value = query_gradient_features(model, batch, need_factors=False, need_full_norm=True)
            norms[group - lo] = float(value["full_norm"])
            if (group - lo + 1) % 512 == 0 or group + 1 == hi:
                norms.flush()
                atomic(out / f"features_progress_rank{rank}.json", dict(
                    phase="actual_TIGER200_full_parameter_norm", done=group - lo + 1,
                    total=hi - lo, elapsed_seconds=time.perf_counter() - began))
        norms.flush()
        torch.cuda.synchronize(device)
        gradient_seconds = time.perf_counter() - began
        if not np.isfinite(norms).all():
            raise RuntimeError("Non-finite actual full-parameter norms")
        hashes = [None] * world
        dist.all_gather_object(hashes, sha(norm_path))
        child = None
        if rank == 0:
            atomic(out / "gradient_ready.json", dict(
                scoring_step=200, world_size=4, total_occurrences=n,
                effective_input_groups=groups, norm_sha256=hashes,
                baseline_protocol_sha256=module.baseline_protocol_sha256,
                index_manifest_sha256=sha(index / "manifest.json"),
                preparation_manifest_sha256=sha(prep / "manifest.json"),
                full_scale_admission_sha256=sha(admission_path),
                selection_not_yet_completed=True))
            repo = Path(__file__).resolve().parents[3]
            command = [sys.executable, "-u", str(repo / "scripts/run_baseline_goracs_selection_cpu.py"),
                       "--root", str(root), "--dataset", module.dataset, "--selection-dir", str(out)]
            log = (out / "CPU_ITRA.log").open("x")
            child = subprocess.Popen(
                command, cwd=repo, env=dict(os.environ, CUDA_VISIBLE_DEVICES="",
                                            OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                                            MKL_NUM_THREADS="4"), stdout=log, stderr=subprocess.STDOUT)
        # Keep collectives short while the isolated CPU solver is working.
        # This does not change optimizer state or run any additional GPU model.
        wait_begin = time.perf_counter()
        signal = torch.zeros(1, dtype=torch.int32, device=device)
        while True:
            if rank == 0:
                code = child.poll()
                signal[0] = 0 if code is None else 1 if code == 0 else -1
            dist.broadcast(signal, src=0)
            result = int(signal.item())
            if result:
                break
            time.sleep(5)
        if rank == 0:
            log.close()
        if result != 1:
            raise RuntimeError("Full-pool CPU ITRA failed; preserve scores/logs")
        CPU_wait_seconds = time.perf_counter() - wait_begin
        result_record = read(out / "cpu_result.json")
        if (result_record["status"] != "completed_actual_full_pool_ITRA"
                or result_record["original_occurrences"] != n
                or result_record["selected_occurrences"] != 4 * n // 5):
            raise RuntimeError("Incomplete actual full-pool ITRA result")
    after = observe()
    cost = dict(
        rank=rank, gradient_device_wall_seconds=gradient_seconds,
        CPU_solver_wait_wall_seconds=CPU_wait_seconds,
        scoring_allocated_rank_wall_seconds=time.perf_counter() - started,
        CPU_seconds=time.process_time() - cpu, state_before=before, state_after=after,
        informational_state_comparison_only=True, norms_sha256=hashes[rank])
    costs = [None] * world
    dist.all_gather_object(costs, cost)
    if rank == 0:
        source = module.asset_manifest
        manifest = dict(
            format_version=1, ready=True, method="GORACS",
            baseline_protocol_sha256=module.baseline_protocol_sha256,
            total_occurrences=n, selected_count=4 * n // 5,
            source_stream_id=source["source_stream_id"],
            replay_shuffle_asset_identity=source["immutable_identity"],
            source_manifest_sha256=sha(module.assets_path / "manifest.json"),
            permutation_sha256=source["files"]["permutation.npy"]["sha256"],
            files={"selected_positions.npy": dict(
                path=str(out / "selected_positions.npy"), sha256=sha(out / "selected_positions.npy"))},
            scoring_step=200, gradient_definition="full trainable TIGER norm; mean four target-token CE",
            reference_source="TRAIN literal pool", reference_count=5000,
            lambda_value=.1, max_improving_exchanges=100, exchange_candidates_each_side=30,
            source_groups_preserve_literal_capacity=True, training_deduplicated=False,
            full_ITRA_executed=True, numerical_prefix_equality_gate=False,
            costs=costs, CPU_selection_cost=dict(
                CPU_seconds=result_record["CPU_seconds"], wall_seconds=result_record["wall_seconds"]),
            cost_accounting="rank wait, allocated scoring, process/wrapper overlap; auxiliary CPU process separate",
            preparation_manifest_sha256=sha(prep / "manifest.json"),
            full_scale_admission_sha256=sha(admission_path),
            CPU_ITRA_result_sha256=sha(out / "cpu_result.json"),
            selection_evidence_sha256={name: sha(out / name) for name in
                                       ("gradient_ready.json", "selected_occurrence_ids.npy",
                                        "selected_group_counts.npy", "cost_matrix.npy")})
        atomic(out / "manifest.json", manifest)
    trainer.strategy.barrier("GORACS_selection_published")
    return out
