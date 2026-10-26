"""Four-GPU runner for stage-dependent spectrum/conflict scheduling."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import open_dict

from scripts import run_data_schedule_ddp as runner


ORIGINAL_BUILD_CONFIG = runner.build_config


def _float_env(name, default):
    return float(os.environ.get(name, str(default)))


def _int_env(name, default):
    return int(os.environ.get(name, str(default)))


EXTRA_PRESENTATION_BUDGETS = {
    "toys": 9_757_428,
    "beauty": 13_434_229,
    "sports": 6_742_922,
}


def build_config(args):
    if (
        args.mode != "native_replay"
        or not args.fast_data
        or args.efficient_eval
        or args.retention != 1.0
        or args.selected_indices
        or args.native_source_dir
        or args.follow_source_trace
        or args.restart_data_stream
    ):
        raise ValueError(
            "Spectrum scheduling requires native_replay, fast-data, full retention, "
            "and a fresh immutable source stream"
        )
    full_assets = os.environ.get("REPLAY_SHUFFLE_ASSETS")
    if not full_assets:
        raise ValueError("Set REPLAY_SHUFFLE_ASSETS to the ready raw TRAIN occurrence stream")

    cfg = ORIGINAL_BUILD_CONFIG(args)
    with open_dict(cfg):
        dm = cfg.data_loading.datamodule
        dm._target_ = (
            "src.data.loading.datamodules.spectrum_conflict_scheduler."
            "SpectrumConflictSchedulerDataModule"
        )
        dm.assets_path = str(Path(full_assets).resolve())
        dm.extra_exposure_budget = _int_env(
            "SCS_EXTRA_PRESENTATION_BUDGET", EXTRA_PRESENTATION_BUDGETS[args.dataset]
        )
        dm.activation_update = _int_env("SCS_ACTIVATION_UPDATE", 200)
        dm.view_readiness_weight = _float_env("SCS_VIEW_READINESS_WEIGHT", 4.0)
        dm.minimum_transition_coverage = _int_env("SCS_MIN_TRANSITION_COVERAGE", 2)
        dm.augmentation_slot_budget = _int_env(
            "SCS_HISTORY_VIEW_BUDGET", int(round(0.75 * EXTRA_PRESENTATION_BUDGETS[args.dataset]))
        )
        dm.augmentation_target_rate = _float_env("SCS_HISTORY_VIEW_RATE", 0.75)
        dm.spectrum_projection_dim = _int_env("SCS_PROJECTION_DIM", 32)
        dm.spectrum_covariance_decay = _float_env("SCS_COVARIANCE_DECAY", 0.99)
        dm.spectrum_ridge = _float_env("SCS_SPECTRUM_RIDGE", 1e-2)
        dm.spectrum_seed = _int_env("SCS_SPECTRUM_SEED", 20261020)
        dm.view_seed = _int_env("SCS_VIEW_SEED", 42 ^ 0xB1E5)
        dm.event_sampling_mode = os.environ.get("SCS_EVENT_SAMPLING_MODE", "spectrum")
        dm.conflict_sampling_mode = os.environ.get("SCS_CONFLICT_SAMPLING_MODE", "conflict")
    full_manifest_path = Path(full_assets).resolve() / "manifest.json"
    output = Path(args.output_dir).resolve()
    if int(os.environ.get("RANK", "0")) == 0:
        output.mkdir(parents=True, exist_ok=True)
        files = [
            "scripts/run_spectrum_scheduled_views.py",
            "scripts/run_data_schedule_ddp.py",
            "src/data/loading/datamodules/spectrum_conflict_scheduler.py",
            "src/data/selection/conflict_aware_presentation.py",
            "src/data/selection/spectrum_event_priority.py",
            "src/data/selection/representation_spectrum_scheduler.py",
        ]
        provenance = {
            "method": "Spectrum-Conflict Scheduler",
            "event_sampling_mode": str(cfg.data_loading.datamodule.event_sampling_mode),
            "conflict_sampling_mode": str(cfg.data_loading.datamodule.conflict_sampling_mode),
            "raw_source": "fixed permutation of original TRAIN occurrences",
            "raw_source_manifest": "manifest.json",
            "raw_train_retention": "all raw TRAIN occurrences retained, including short histories",
            "augmentation_eligibility": "history_length >= 2 and target > history_length",
            "transition_coverage_definition": "number of distinct original TRAIN users sharing the complete-SID adjacent transition",
            "minimum_transition_coverage": int(cfg.data_loading.datamodule.minimum_transition_coverage),
            "augmentation_target_rate": float(cfg.data_loading.datamodule.augmentation_target_rate),
            "augmentation_slot_budget": int(cfg.data_loading.datamodule.augmentation_slot_budget),
            "ineligible_extra_action": "literal repeat",
            "uses_precomputed_view_assets": False,
            "uses_relation_manifest": False,
            "uses_relation_assets": False,
            "uses_probe": False,
            "uses_old_checkpoint": False,
            "fixed_transform": "ordered earlier-history view; target, length, and latest item preserved",
            "activation_update": int(cfg.data_loading.datamodule.activation_update),
            "extra_exposure_budget": int(cfg.data_loading.datamodule.extra_exposure_budget),
            "budget_source": "domain-specific matched additional-presentation budget",
            "extra_exposure_position": "selected online and appended to the currently consumed raw batch",
            "view_readiness_weight": float(cfg.data_loading.datamodule.view_readiness_weight),
            "event_selection": (
                "online spectrum distribution combined with the transition-coverage candidate pool and view-readiness weight"
                if cfg.data_loading.datamodule.event_sampling_mode == "spectrum"
                else "uniform distribution over the same transition-coverage candidate pool and view-readiness weights"
            ),
            "action_selection": (
                "fixed scalar augmentation quota; event slots use the configured event sampler; view slots additionally use output-head gradient-conflict action probability"
                if cfg.data_loading.datamodule.conflict_sampling_mode == "conflict"
                else "fixed scalar view quota; event slots use the configured event sampler; view/repeat allocation is uniform over view-ready events"
            ),
            "source_manifests": {
                "full_sha256": hashlib.sha256(full_manifest_path.read_bytes()).hexdigest(),
            },
            "scheduler": {
                "projection_dim": int(cfg.data_loading.datamodule.spectrum_projection_dim),
                "covariance_decay": float(cfg.data_loading.datamodule.spectrum_covariance_decay),
                "ridge": float(cfg.data_loading.datamodule.spectrum_ridge),
                "spectrum_seed": int(cfg.data_loading.datamodule.spectrum_seed),
                "view_seed": int(cfg.data_loading.datamodule.view_seed),
            },
            "source_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files
            },
        }
        (output / "spectrum_provenance.json").write_text(json.dumps(provenance, indent=2))
    return cfg


if __name__ == "__main__":
    runner.build_config = build_config
    runner.main()

