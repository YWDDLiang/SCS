"""Stage-dependent spectrum and history-view scheduling from raw TRAIN.

The base stream is a fixed permutation of raw TRAIN occurrences. At transfer
time the scheduler scores the current raw microbatch, samples
additional events from that microbatch, and appends either an exact repeat or
a freshly generated ordered-history view.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader, IterableDataset

from src.data.loading.components.cached_eval_batches import _identity
from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.data.loading.datamodules.ddp_scheduled_datamodule import (
    DDPScheduledSequenceDataModule,
)
from src.data.loading.datamodules.replay_shuffle_datamodule import (
    ReplayShuffleSequenceDataModule,
)
from src.data.selection.conflict_aware_presentation import ConflictAwarePresentationPolicy
from src.data.selection.spectrum_event_priority import (
    augmentation_eligibility,
    cross_user_transition_coverage,
    coverage_aware_spectrum_probabilities,
    partition_integer_budget,
    sample_scheduled_presentations,
    coverage_for_events,
)
from src.data.selection.representation_spectrum_scheduler import RepresentationSpectrumScheduler


class _RawSpectrumBatches(IterableDataset):
    """Four-rank sharding of raw P1 batches; no auxiliary slots are prebuilt."""

    def __init__(self, module):
        self.module = module

    def __iter__(self):
        module = self.module
        cursor = module.consumed_batches
        rank = module.trainer.global_rank
        while True:
            for index in range(cursor + rank, cursor + 128, 4):
                yield module.occurrences_at(index), {
                    "spectrum_global_batch_index": int(index)
                }
            cursor += 128


class SpectrumConflictSchedulerDataModule(ReplayShuffleSequenceDataModule):
    """Prioritize raw training events and allocate additional history views."""

    def __init__(
        self,
        *args,
        extra_exposure_budget=0,
        activation_update=200,
        view_readiness_weight=4.0,
        minimum_transition_coverage=2,
        augmentation_slot_budget=5_038_053,
        augmentation_target_rate=0.5314753973153495,
        spectrum_projection_dim=32,
        spectrum_covariance_decay=0.99,
        spectrum_ridge=1e-2,
        spectrum_seed=20261020,
        spectrum_representation_scope="encoder_mean",
        view_seed=42 ^ 0xB1E5,
        conflict_ema_decay=0.90,
        conflict_min_view_rate=0.50,
        conflict_max_view_rate=0.90,
        event_sampling_mode="spectrum",
        conflict_sampling_mode="conflict",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if int(extra_exposure_budget) < 0:
            raise ValueError("extra_exposure_budget must be non-negative")
        if int(activation_update) <= 0:
            raise ValueError("activation_update must be positive")
        if not np.isfinite(float(view_readiness_weight)) or float(view_readiness_weight) <= 0:
            raise ValueError("view_readiness_weight must be finite and positive")
        self.extra_exposure_budget = int(extra_exposure_budget)
        self.activation_update = int(activation_update)
        self.view_readiness_weight = float(view_readiness_weight)
        self.minimum_transition_coverage = int(minimum_transition_coverage)
        self.augmentation_slot_budget = int(augmentation_slot_budget)
        self.augmentation_target_rate = float(augmentation_target_rate)
        if self.minimum_transition_coverage < 1:
            raise ValueError("minimum_transition_coverage must be positive")
        if not np.isfinite(self.augmentation_target_rate) or not 0.0 <= self.augmentation_target_rate <= 1.0:
            raise ValueError("augmentation_target_rate must lie in [0, 1]")
        if self.augmentation_slot_budget < 0 or self.augmentation_slot_budget > self.extra_exposure_budget:
            raise ValueError("augmentation_slot_budget must lie within the extra exposure budget")
        self.spectrum = RepresentationSpectrumScheduler(
            projection_dim=int(spectrum_projection_dim),
            covariance_decay=float(spectrum_covariance_decay),
            ridge=float(spectrum_ridge),
            target_view_rate=0.0,
            seed=int(spectrum_seed),
        )
        self.spectrum_representation_scope = str(spectrum_representation_scope)
        if self.spectrum_representation_scope not in ("encoder_mean", "suffix_last2"):
            raise ValueError("spectrum_representation_scope must be encoder_mean or suffix_last2")
        self.policy = ConflictAwarePresentationPolicy(
            temperature=1.0,
            min_augmentation_rate=float(conflict_min_view_rate),
            max_augmentation_rate=float(conflict_max_view_rate),
            seed=int(spectrum_seed),
        )
        self.view_seed = int(view_seed)
        self.conflict_ema_decay = float(conflict_ema_decay)
        self.conflict_min_view_rate = float(conflict_min_view_rate)
        self.conflict_max_view_rate = float(conflict_max_view_rate)
        self.event_sampling_mode = str(event_sampling_mode)
        self.conflict_sampling_mode = str(conflict_sampling_mode)
        if self.event_sampling_mode not in ("spectrum", "uniform"):
            raise ValueError("event_sampling_mode must be spectrum or uniform")
        if self.conflict_sampling_mode not in ("conflict", "uniform"):
            raise ValueError("conflict_sampling_mode must be conflict or uniform")
        if not 0.0 <= self.conflict_min_view_rate <= self.conflict_max_view_rate <= 1.0:
            raise ValueError("conflict view-rate bounds must lie in [0, 1]")
        self.conflict_reference = None
        self.conflict_last_rate = 0.0
        self.conflict_last_target_rate = 0.0
        self.spectrum_action_counts = np.zeros(2, dtype=np.int64)
        self.spectrum_effective_rank = 0.0
        self.spectrum_last_step = -1
        self.extra_exposure_used = 0
        self.extra_credit = 0.0
        self.rank_budget = None
        self.rank_augmentation_slot_budget = None
        self.rank_credit = 0.0
        self.augmentation_credit = 0.0
        self.transition_coverage_by_user_target = None
        self.coverage_sorted_user_ids = None
        self.user_rows_in_coverage_order = None
        self.single_user_transition_candidates = 0
        self.coverage_qualified_candidates = 0
        self._rng = np.random.default_rng(int(spectrum_seed) + int(os.environ.get("RANK", "0")))
        self.spectrum_trace = Path(".") / (
            f"spectrum_schedule_rank{int(os.environ.get('RANK', '0'))}.jsonl"
        )
        self.exposure = {
            "batches": 0,
            "samples": 0,
            "base_samples": 0,
            "extra_exposure": 0,
            "augmentation_eligible_rows_scored": 0,
            "eligible_extra_exposures": 0,
            "forced_repeat_exposures": 0,
            "single_user_extra_exposures": 0,
            "coverage_qualified_extra_exposures": 0,
            "auxiliary_slots": 0,
            "copies": 0,
            "context_changes": 0,
            "generated_changed": 0,
            "generated_novel_in_source": 0,
            "spectrum_scored_rows": 0,
            "selection_events": 0,
        }

    def train_dataloader(self):
        self._require_world()
        if self.consumed_batches % self.global_accumulation_batches:
            raise RuntimeError("Raw spectrum stream resumes only at an optimizer boundary")
        self._identity_value = self._source_identity()
        config = self.stage_to_config[TrainerFn.FITTING]
        from src.data.selection.subsequence_candidate import SubsequenceCandidateStore
        from src.data.loading.components.fast_scheduled_collate import FastScheduledCollator

        self.store = SubsequenceCandidateStore.load(self.candidate_store_path)
        if self.store.sequence_length != config.sequence_length:
            raise ValueError("Candidate sequence length mismatch")
        self._fast_collate = FastScheduledCollator(
            self.store,
            labels=config.labels,
            sequence_length=config.sequence_length,
            masking_token=config.masking_token,
            padding_token=config.padding_token,
            oov_token=config.get("oov_token", None),
        )
        self.transition_coverage_by_user_target = cross_user_transition_coverage(self.store)
        user_ids = self.store.user_ids.cpu().numpy()
        self.user_rows_in_coverage_order = np.argsort(user_ids, kind="stable")
        self.coverage_sorted_user_ids = user_ids[self.user_rows_in_coverage_order]
        triples = self.store.candidate_triples.cpu().numpy()
        candidate_eligible = augmentation_eligibility(triples)
        candidate_coverage = self.transition_coverage_by_user_target[
            triples[:, 0], triples[:, 2]
        ]
        self.single_user_transition_candidates = int((candidate_coverage == 1).sum())
        self.coverage_qualified_candidates = int(
            (candidate_eligible & (candidate_coverage >= self.minimum_transition_coverage)).sum()
        )
        if np.any(candidate_coverage <= 0):
            raise RuntimeError("raw TRAIN transition table does not cover every candidate event")
        rank = int(self.trainer.global_rank)
        world_size = int(self.trainer.world_size)
        self.rank_budget = int(partition_integer_budget(self.extra_exposure_budget, world_size)[rank])
        self.rank_augmentation_slot_budget = int(
            partition_integer_budget(self.augmentation_slot_budget, world_size)[rank]
        )
        self.rank_credit = 0.0
        self.spectrum_trace = Path(self.trainer.default_root_dir).resolve() / (
            f"online/spectrum_schedule_rank{rank}.jsonl"
        )
        self.spectrum_trace.parent.mkdir(parents=True, exist_ok=True)
        if rank == 0 and not self.spectrum_trace.exists():
            self.spectrum_trace.write_text("")
        return (
            DataLoader(
                _RawSpectrumBatches(self),
                batch_size=None,
                num_workers=0,
                pin_memory=config.pin_memory,
                collate_fn=_identity,
                generator=torch.Generator().manual_seed(self.stream_seed),
            ),
        )

    @staticmethod
    def _representation(model, batch, scope="encoder_mean"):
        """Return the online spectral representation for the current raw view."""
        model_input = batch[0]
        sequence = model_input.transformed_sequences["sequence_data"]
        mask = model_input.mask
        device = next(model.parameters()).device
        sequence = sequence.to(device=device, dtype=torch.long)
        mask = mask.to(device=device, dtype=torch.long)
        if not hasattr(model, "get_embedding_table") or not hasattr(
            model, "_add_repeating_offset_to_rows"
        ):
            raise TypeError("Spectrum scheduling requires the TIGER encoder embedding API")
        shifted = model._add_repeating_offset_to_rows(
            input_sids=sequence,
            codebook_size=model.num_embeddings_per_hierarchy,
            num_hierarchies=model.num_hierarchies,
            attention_mask=mask,
        )
        with torch.no_grad():
            embeddings = model.get_embedding_table(table_name="encoder")(shifted)
            if scope == "encoder_mean":
                weights = mask.to(dtype=embeddings.dtype).unsqueeze(-1)
            elif scope == "suffix_last2":
                # The collator's mask covers the history after the target label
                # is removed. Restrict the signal to the last two complete
                # history items used for transition coverage computation.
                hierarchy = int(model.num_hierarchies)
                valid_lengths = mask.sum(dim=1).to(dtype=torch.long)
                history_end = valid_lengths.clamp_min(0)
                history_start = (history_end - 2 * hierarchy).clamp_min(0)
                positions = torch.arange(sequence.shape[1], device=device).unsqueeze(0)
                suffix_mask = (
                    (positions >= history_start.unsqueeze(1))
                    & (positions < history_end.unsqueeze(1))
                )
                weights = suffix_mask.to(dtype=embeddings.dtype).unsqueeze(-1)
            else:
                raise ValueError(
                    "scope must be encoder_mean or suffix_last2"
                )
            return (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _move_batch(model_input, label_data, device):
        model_input.mask = model_input.mask.to(device)
        model_input.transformed_sequences = {
            key: value.to(device) for key, value in model_input.transformed_sequences.items()
        }
        label_data.labels = {
            key: value.to(device) for key, value in label_data.labels.items()
        }
        label_data.label_location = {
            key: value.to(device) for key, value in label_data.label_location.items()
        }
        label_data.attention_mask = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in label_data.attention_mask.items()
        }
        return model_input, label_data

    def _output_head_conflict(self, model, batch):
        """No-backward per-example output-head gradient proxy."""
        model_input, label_data = batch
        device = next(model.parameters()).device
        model_input, label_data = self._move_batch(model_input, label_data, device)
        hierarchy = int(model.num_hierarchies)
        targets = label_data.labels["sequence_data"].reshape(
            model_input.mask.shape[0], hierarchy
        ).long()
        inputs = {
            model.feature_to_model_input_map.get(name, name): value
            for name, value in model_input.transformed_sequences.items()
        }
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                hidden = model(
                    attention_mask_encoder=model_input.mask,
                    future_ids=targets,
                    **inputs,
                )[:, :-1].float()
                logits = torch.stack(
                    [model.decoder.decoder_mlp[level](hidden[:, level]) for level in range(hierarchy)],
                    dim=1,
                ).float()
                errors = logits.softmax(dim=-1)
                errors.scatter_add_(
                    2,
                    targets.unsqueeze(-1),
                    -torch.ones_like(targets, dtype=errors.dtype).unsqueeze(-1),
                )
                current = [
                    torch.einsum("bv,bh->vh", errors[:, level], hidden[:, level])
                    / max(1, hidden.shape[0])
                    for level in range(hierarchy)
                ]
                if self.conflict_reference is None:
                    conflict = torch.zeros(hidden.shape[0], device=device)
                else:
                    reference = [value.to(device=device, dtype=hidden.dtype) for value in self.conflict_reference]
                    dot = sum(
                        torch.einsum(
                            "bv,vh,bh->b", errors[:, level], reference[level], hidden[:, level]
                        )
                        for level in range(hierarchy)
                    )
                    candidate_norm_sq = sum(
                        errors[:, level].square().sum(dim=-1)
                        * hidden[:, level].square().sum(dim=-1)
                        for level in range(hierarchy)
                    )
                    reference_norm = torch.sqrt(
                        sum(value.square().sum() for value in reference).clamp_min(1e-12)
                    )
                    cosine = dot / (
                        torch.sqrt(candidate_norm_sq).clamp_min(1e-12) * reference_norm
                    )
                    conflict = (-cosine).clamp_min(0.0)
                decay = self.conflict_ema_decay
                if self.conflict_reference is None:
                    self.conflict_reference = [value.detach().cpu().clone() for value in current]
                else:
                    self.conflict_reference = [
                        (old.to(device=device, dtype=value.dtype) * decay + value * (1.0 - decay))
                        .detach()
                        .cpu()
                        for old, value in zip(self.conflict_reference, current)
                    ]
        finally:
            model.train(was_training)
        return conflict.detach()

    @staticmethod
    def _row_seed(seed, ordinal, row):
        value = int(seed) & ((1 << 63) - 1)
        for part in (int(ordinal) + 1, *(int(v) + 1 for v in row)):
            value ^= (int(part) * 0x9E3779B185EBCA87) & ((1 << 63) - 1)
            value = ((value << 13) | (value >> 50)) & ((1 << 63) - 1)
        return value

    def _make_view_positions(self, rows, ordinals, augment):
        rows = np.asarray(rows, dtype=np.int64)
        augment = np.asarray(augment, dtype=bool)
        if augment.shape != (len(rows),):
            raise ValueError("view-action mask must align with selected raw events")
        positions = np.full((len(rows), 29), 65535, dtype=np.uint16)
        eligible = augmentation_eligibility(rows)
        for local, (row, ordinal) in enumerate(zip(rows, ordinals)):
            _, start, target = (int(value) for value in row)
            history_length = target - start
            length = min(history_length, 29)
            if length <= 0:
                continue
            positions[local, :length] = np.arange(target - length, target, dtype=np.uint16)
            if not eligible[local] or not augment[local]:
                continue
            generator = np.random.default_rng(self._row_seed(self.view_seed, ordinal, row))
            candidates = np.arange(target - 1, dtype=np.int64)
            original_prefix = np.arange(target - length, target - 1, dtype=np.int64)
            selected = np.sort(generator.choice(candidates, size=length - 1, replace=False))
            if np.array_equal(selected, original_prefix):
                selected = np.arange(length - 1, dtype=np.int64)
            positions[local, : length - 1] = selected.astype(np.uint16)
            positions[local, length - 1] = np.uint16(target - 1)
        return positions, eligible

    def _next_extra_count(self, base_count):
        if self.rank_budget is None or self.extra_exposure_used >= self.rank_budget:
            return 0
        rate = self.extra_exposure_budget / max(1, self.occurrence_count)
        self.rank_credit += rate * int(base_count)
        requested = int(self.rank_credit)
        self.rank_credit -= requested
        return min(requested, self.rank_budget - self.extra_exposure_used)

    def _scheduled_details(self, index, model, rows):
        rows = np.asarray(rows, dtype=np.int32).copy()
        base_count = len(rows)
        details = {
            "base_rows": rows,
            "rows": rows,
            "changed": np.zeros(base_count, dtype=bool),
            "view_positions": np.full((base_count, 29), 65535, dtype=np.uint16),
            "ids": np.asarray(self.last_occurrence_indices, dtype=np.int64).copy(),
            "extra_indices": np.empty(0, dtype=np.int64),
            "selected_probabilities": np.empty(0, dtype=np.float32),
            "augmentation_eligible_candidates": 0,
            "eligible_sampling_probability_mass": 0.0,
            "eligible_extra_exposures": 0,
            "forced_repeat_exposures": 0,
            "single_user_extra_exposures": 0,
            "coverage_qualified_extra_exposures": 0,
        }
        if int(self.trainer.global_step) < self.activation_update:
            self.exposure["base_samples"] += base_count
            self.exposure["samples"] += base_count
            return details
        result = None
        if self.event_sampling_mode == "spectrum":
            raw_batch = self._fast_collate(torch.from_numpy(rows))
            representations = self._representation(
                model, raw_batch, self.spectrum_representation_scope
            )
            result = self.spectrum.score_and_observe(representations)
            spectrum_probabilities = self.policy.event_probabilities(result.scores).detach().cpu().numpy()
        else:
            # Uniform control: keep the same candidate pool, view readiness,
            # weighting, budgets, and conflict policy, but remove spectrum from
            # event identity selection entirely.
            spectrum_probabilities = np.full(
                base_count, 1.0 / max(1, base_count), dtype=np.float64
            )
        eligible_rows = augmentation_eligibility(rows)
        if self.transition_coverage_by_user_target is None:
            raise RuntimeError("raw TRAIN transition table was not initialized")
        coverage_rows = coverage_for_events(
            rows,
            self.store,
            self.transition_coverage_by_user_target,
            sorted_user_ids=self.coverage_sorted_user_ids,
            user_rows_in_sorted_order=self.user_rows_in_coverage_order,
        )
        coverage_qualified = coverage_rows >= self.minimum_transition_coverage
        gated_rows = eligible_rows & coverage_qualified
        probabilities = coverage_aware_spectrum_probabilities(
            spectrum_probabilities,
            coverage_rows,
            eligible_rows,
            minimum_coverage=self.minimum_transition_coverage,
            view_readiness_weight=self.view_readiness_weight,
        )
        probability_mass = probabilities.sum()
        if probability_mass <= 0:
            # Keep the raw batch and defer the fractional exposure credit.
            # A rare batch without a qualified event must not stop training.
            self.rank_credit += self._next_extra_count(base_count)
            self.exposure["base_samples"] += base_count
            self.exposure["samples"] += base_count
            if self.event_sampling_mode == "spectrum":
                self.exposure["spectrum_scored_rows"] += base_count
            self._write_trace(
                index,
                result,
                conflict_rate=0.0,
                target_rate=0.0,
                probabilities=spectrum_probabilities,
                details=details,
            )
            return details
        probabilities = probabilities / probability_mass
        details["eligible_sampling_probability_mass"] = float(probabilities[eligible_rows].sum())
        eligible_indices = np.flatnonzero(gated_rows)
        details["augmentation_eligible_candidates"] = int(len(eligible_indices))
        action_probability_by_row = np.zeros(base_count, dtype=np.float32)
        conflict_rate = 0.0
        target_rate = 0.0
        if len(eligible_indices) and self.conflict_sampling_mode == "conflict":
            eligible_batch = self._fast_collate(
                torch.from_numpy(rows[eligible_indices].copy())
            )
            conflict_cpu = self._output_head_conflict(model, eligible_batch).detach().cpu()
            eligible_action_probabilities, target_rate = self.policy.augmentation_probabilities(
                conflict_cpu,
                torch.from_numpy(probabilities[eligible_indices]),
            )
            action_probability_by_row[eligible_indices] = (
                eligible_action_probabilities.numpy()
            )
            conflict_rate = float((conflict_cpu > 0).float().mean())
        elif len(eligible_indices):
            # Uniform control: preserve the exact global view-slot quota while
            # making all coverage-qualified, view-ready rows equally likely.
            action_probability_by_row[eligible_indices] = 1.0
            target_rate = float(self.augmentation_target_rate)
        self.conflict_last_rate = conflict_rate
        self.conflict_last_target_rate = float(target_rate)
        self.exposure["augmentation_eligible_rows_scored"] += len(eligible_indices)
        extra_count = self._next_extra_count(base_count)
        if extra_count:
            desired = self.augmentation_credit + self.augmentation_target_rate * extra_count
            remaining_augmentation_slots = max(
                0, self.rank_augmentation_slot_budget - int(self.spectrum_action_counts[1])
            )
            augmentation_count = min(extra_count, int(np.floor(desired)), remaining_augmentation_slots)
            self.augmentation_credit = desired - augmentation_count
            if augmentation_count and not len(eligible_indices):
                self.augmentation_credit += augmentation_count
                augmentation_count = 0
            selected, augment = sample_scheduled_presentations(
                probabilities=probabilities,
                augmentation_probabilities=action_probability_by_row,
                view_eligible=gated_rows,
                extra_count=extra_count,
                augmentation_count=augmentation_count,
                rng=self._rng,
                candidate_mask=coverage_qualified,
            )
            extra_rows = rows[selected].copy()
            extra_ordinals = np.arange(
                self.extra_exposure_used,
                self.extra_exposure_used + extra_count,
                dtype=np.int64,
            )
            positions, view_eligible = self._make_view_positions(extra_rows, extra_ordinals, augment)
            if not np.array_equal(view_eligible, eligible_rows[selected]):
                raise RuntimeError("Augmentation eligibility changed between selection and view creation")
            if np.any(coverage_rows[selected] < self.minimum_transition_coverage):
                raise RuntimeError("Spectrum scheduler selected an event outside its candidate pool")
            details["rows"] = np.concatenate([rows, extra_rows], axis=0)
            details["changed"] = np.concatenate(
                [np.zeros(base_count, dtype=bool), augment], axis=0
            )
            details["view_positions"] = np.concatenate(
                [details["view_positions"], positions], axis=0
            )
            details["extra_indices"] = selected.astype(np.int64)
            details["selected_probabilities"] = probabilities[selected].astype(np.float32)
            details["eligible_extra_exposures"] = int(gated_rows[selected].sum())
            details["forced_repeat_exposures"] = int((~gated_rows[selected]).sum())
            details["single_user_extra_exposures"] = int((coverage_rows[selected] == 1).sum())
            details["coverage_qualified_extra_exposures"] = int(
                coverage_qualified[selected].sum()
            )
            self.extra_exposure_used += extra_count
            self.spectrum_action_counts += np.bincount(augment.astype(np.int64), minlength=2)
            self.exposure["extra_exposure"] += extra_count
            self.exposure["auxiliary_slots"] += extra_count
            self.exposure["eligible_extra_exposures"] += details["eligible_extra_exposures"]
            self.exposure["forced_repeat_exposures"] += details["forced_repeat_exposures"]
            self.exposure["single_user_extra_exposures"] += details["single_user_extra_exposures"]
            self.exposure["coverage_qualified_extra_exposures"] += details["coverage_qualified_extra_exposures"]
            self.exposure["copies"] += int((~augment).sum())
            self.exposure["context_changes"] += int(augment.sum())
            self.exposure["generated_changed"] += int(augment.sum())
            self.exposure["selection_events"] += extra_count
        self.exposure["base_samples"] += base_count
        self.exposure["samples"] += len(details["rows"])
        if self.event_sampling_mode == "spectrum":
            self.exposure["spectrum_scored_rows"] += base_count
        self._write_trace(index, result, conflict_rate, target_rate, probabilities, details)
        return details

    def _write_trace(self, index, result, conflict_rate, target_rate, probabilities, details):
        rank = int(os.environ.get("RANK", "0"))
        step = int(self.trainer.global_step)
        if rank != 0 or step == self.spectrum_last_step:
            return
        self.spectrum_last_step = step
        self.spectrum_trace.parent.mkdir(parents=True, exist_ok=True)
        with self.spectrum_trace.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "update": step,
                        "global_batch": int(index),
                        "scored_rows": int(len(probabilities)),
                        "extra_exposures": int(len(details["extra_indices"])),
                        "changed_exposures": int(details["changed"].sum()),
                        "repeat_exposures": int(len(details["extra_indices"]) - details["changed"].sum()),
                        "augmentation_eligible_candidates": int(
                            details["augmentation_eligible_candidates"]
                        ),
                        "eligible_sampling_probability_mass": float(
                            details["eligible_sampling_probability_mass"]
                        ),
                        "view_readiness_weight": self.view_readiness_weight,
                        "eligible_extra_exposures": int(details["eligible_extra_exposures"]),
                        "forced_repeat_exposures": int(details["forced_repeat_exposures"]),
                        "single_user_extra_exposures": int(details["single_user_extra_exposures"]),
                        "coverage_qualified_extra_exposures": int(details["coverage_qualified_extra_exposures"]),
                        "effective_rank": (
                            float(result.effective_rank.detach().cpu())
                            if result is not None else 0.0
                        ),
                        "mean_spectral_residual": (
                            float(result.scores.mean().detach().cpu())
                            if result is not None else 0.0
                        ),
                        "max_spectral_residual": (
                            float(result.scores.max().detach().cpu())
                            if result is not None else 0.0
                        ),
                        "mean_selection_probability": float(probabilities.mean()),
                        "gradient_conflict_rate": float(conflict_rate),
                        "target_augmentation_rate": float(target_rate),
                        "fixed_augmentation_target_rate": self.augmentation_target_rate,
                    },
                    allow_nan=False,
                )
                + "\n"
            )

    def _collate_dynamic_batch(self, details):
        rows = torch.from_numpy(np.asarray(details["rows"], dtype=np.int64))
        if not details["changed"].any():
            return self._fast_collate(rows)
        collator = self._fast_collate
        uids, starts, targets = rows.unbind(1)
        locations = torch.searchsorted(collator.sorted_uids, uids.contiguous())
        if torch.any(locations >= len(collator.sorted_uids)) or not torch.equal(
            collator.sorted_uids[locations], uids
        ):
            raise KeyError("View source user absent from original TRAIN store")
        store_rows = collator.uid_rows[locations]
        hierarchy = self.store.sid_hierarchy
        if hierarchy != 4 or collator.sequence_length != 120:
            raise ValueError("Ordered history views require four SID tokens and length 120")
        lengths = (targets - starts).clamp(max=29)
        item_positions = (targets - lengths)[:, None] + torch.arange(30)[None, :]
        changed = torch.from_numpy(details["changed"])
        frozen = torch.from_numpy(np.asarray(details["view_positions"], dtype=np.int64))
        valid = torch.arange(29)[None, :] < lengths[:, None]
        item_positions[:, :29] = torch.where(
            changed[:, None] & valid, frozen[:, :29], item_positions[:, :29]
        )
        item_positions[torch.arange(len(rows)), lengths] = targets
        indices = (
            self.store.offsets[store_rows, None, None]
            + item_positions[:, :, None] * hierarchy
            + torch.arange(hierarchy)[None, None, :]
        ).reshape(len(rows), 120).clamp(max=self.store.tokens.numel() - 1)
        sequences = self.store.tokens[indices].to(torch.int64)
        positions = collator.positions[None, :]
        sequences.masked_fill_(
            positions >= ((lengths + 1) * hierarchy)[:, None], collator.padding_token
        )
        user_ids = torch.full_like(sequences, collator.padding_token)
        user_ids[:, 0] = uids
        model_input, model_labels = SequentialModelInputData(), SequentialModuleLabelData()
        for field, values in (("sequence_data", sequences), ("user_id", user_ids)):
            if field in collator.labels:
                transformed = collator.labels[field].transform.transform_label(
                    sequence=values,
                    padding_token=collator.padding_token,
                    masking_token=collator.masking_token,
                )
                model_input.transformed_sequences[field] = transformed.sequence
                model_labels.labels[field] = transformed.labels
                model_labels.label_location[field] = transformed.label_location
                model_labels.attention_mask[field] = transformed.attention_mask
            else:
                model_input.transformed_sequences[field] = values
            if model_input.mask is None:
                model_input.mask = (values != collator.padding_token).long()
        return model_input, model_labels

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.training:
            return batch
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[1], dict):
            meta = batch[1]
            if set(meta) == {"spectrum_global_batch_index"}:
                index = int(meta["spectrum_global_batch_index"])
                rank = int(self.trainer.global_rank)
                if index % 4 != rank or index - rank != self.consumed_batches:
                    raise RuntimeError("Spectrum raw rank/global cursor mismatch")
                details = self._scheduled_details(
                    index, self.trainer.lightning_module, batch[0].numpy()
                )
                self.exposure["batches"] += 1
                self.last_occurrence_indices = details["ids"].copy()
                output = self._collate_dynamic_batch(details)
                return DDPScheduledSequenceDataModule.on_before_batch_transfer(
                    self, (*output, {"ddp_global_batch_index": index}), dataloader_idx
                )
        if isinstance(batch, (tuple, list)):
            children = [self.on_before_batch_transfer(child, dataloader_idx) for child in batch]
            return tuple(children) if isinstance(batch, tuple) else children
        return batch

    def state_dict(self):
        state = super().state_dict()
        state.update(
            spectrum_scheduler_format_version=10,
            spectrum_scheduler=self.spectrum.state_dict(),
            spectrum_policy=self.policy.state_dict(),
            spectrum_action_counts=self.spectrum_action_counts.copy(),
            spectrum_effective_rank=self.spectrum_effective_rank,
            spectrum_last_step=self.spectrum_last_step,
            conflict_reference=self.conflict_reference,
            conflict_last_rate=self.conflict_last_rate,
            conflict_last_target_rate=self.conflict_last_target_rate,
            extra_exposure_budget=self.extra_exposure_budget,
            view_readiness_weight=self.view_readiness_weight,
            minimum_transition_coverage=self.minimum_transition_coverage,
            event_sampling_mode=self.event_sampling_mode,
            conflict_sampling_mode=self.conflict_sampling_mode,
            augmentation_target_rate=self.augmentation_target_rate,
            augmentation_slot_budget=self.augmentation_slot_budget,
            spectrum_representation_scope=self.spectrum_representation_scope,
            augmentation_credit=self.augmentation_credit,
            extra_exposure_used=self.extra_exposure_used,
            extra_credit=self.extra_credit,
            rank_budget=self.rank_budget,
            rank_augmentation_slot_budget=self.rank_augmentation_slot_budget,
            rank_credit=self.rank_credit,
            spectrum_exposure=dict(self.exposure),
        )
        return state

    def load_state_dict(self, state):
        if getattr(getattr(self.trainer, "state", None), "fn", None) in (
            TrainerFn.TESTING,
            TrainerFn.VALIDATING,
            TrainerFn.PREDICTING,
        ):
            return
        if state.get("spectrum_scheduler_format_version") != 10:
            raise RuntimeError("checkpoint lacks the raw-stream spectrum scheduler state")
        super().load_state_dict(state)
        if int(state["extra_exposure_budget"]) != self.extra_exposure_budget:
            raise RuntimeError("extra exposure budget differs from checkpoint")
        if float(state["view_readiness_weight"]) != self.view_readiness_weight:
            raise RuntimeError("view candidate weight differs from checkpoint")
        if int(state["minimum_transition_coverage"]) != self.minimum_transition_coverage:
            raise RuntimeError("minimum transition coverage differs from checkpoint")
        if state.get("event_sampling_mode", "spectrum") != self.event_sampling_mode:
            raise RuntimeError("event sampling mode differs from checkpoint")
        if state.get("conflict_sampling_mode", "conflict") != self.conflict_sampling_mode:
            raise RuntimeError("conflict sampling mode differs from checkpoint")
        if float(state["augmentation_target_rate"]) != self.augmentation_target_rate:
            raise RuntimeError("augmentation target rate differs from checkpoint")
        if int(state["augmentation_slot_budget"]) != self.augmentation_slot_budget:
            raise RuntimeError("augmentation slot budget differs from checkpoint")
        if state.get("spectrum_representation_scope", "encoder_mean") != self.spectrum_representation_scope:
            raise RuntimeError("spectrum representation scope differs from checkpoint")
        if int(state["rank_augmentation_slot_budget"]) != self.rank_augmentation_slot_budget:
            raise RuntimeError("rank-local augmentation slot budget differs from checkpoint")
        self.spectrum.load_state_dict(state["spectrum_scheduler"])
        self.policy.load_state_dict(state["spectrum_policy"])
        self.spectrum_action_counts = np.asarray(state["spectrum_action_counts"], dtype=np.int64)
        self.spectrum_effective_rank = float(state["spectrum_effective_rank"])
        self.spectrum_last_step = int(state["spectrum_last_step"])
        self.conflict_reference = state.get("conflict_reference")
        self.conflict_last_rate = float(state.get("conflict_last_rate", 0.0))
        self.conflict_last_target_rate = float(state.get("conflict_last_target_rate", 0.0))
        self.extra_exposure_used = int(state["extra_exposure_used"])
        self.augmentation_credit = float(state["augmentation_credit"])
        self.extra_credit = float(state["extra_credit"])
        self.rank_budget = state.get("rank_budget")
        self.rank_credit = float(state.get("rank_credit", 0.0))
        if state["spectrum_exposure"] != self.exposure:
            raise RuntimeError("saved spectrum exposure counters disagree")

    def method_summary(self):
        device = self.trainer.strategy.root_device
        counts = torch.as_tensor(self.spectrum_action_counts, dtype=torch.int64, device=device)
        values = torch.tensor(
            [
                self.extra_exposure_used,
                self.exposure.get("base_samples", 0),
                self.spectrum.observed,
                self.exposure.get("augmentation_eligible_rows_scored", 0),
                self.exposure.get("eligible_extra_exposures", 0),
                self.exposure.get("forced_repeat_exposures", 0),
                self.exposure.get("single_user_extra_exposures", 0),
                self.exposure.get("coverage_qualified_extra_exposures", 0),
            ],
            dtype=torch.int64,
            device=device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        total_extra = int(values[0].cpu())
        total_augmented = int(counts[1].cpu())
        if total_augmented > self.augmentation_slot_budget:
            raise RuntimeError("global augmentation slot budget was exceeded")
        total_base = int(values[1].cpu())
        return {
            "method": "Online scheduling from raw TRAIN",
            "event_sampling_mode": self.event_sampling_mode,
            "conflict_sampling_mode": self.conflict_sampling_mode,
            "action_names": ["repeat", "ordered_history_view"],
            "action_counts": counts.cpu().tolist(),
            "extra_exposure_budget": self.extra_exposure_budget,
            "view_readiness_weight": self.view_readiness_weight,
            "extra_exposure_used": total_extra,
            "extra_exposure_remaining": max(0, self.extra_exposure_budget - total_extra),
            "actual_augmentation_rate": float(counts[1].cpu()) / max(1, total_extra),
            "actual_repeat_rate": float(counts[0].cpu()) / max(1, total_extra),
            "base_samples_scored": total_base,
            "augmentation_eligibility": "history_length >= 2 and target > history_length",
            "augmentation_eligible_rows_scored": int(values[3].cpu()),
            "eligible_extra_exposures": int(values[4].cpu()),
            "forced_repeat_exposures": int(values[5].cpu()),
            "single_user_extra_exposures": int(values[6].cpu()),
            "coverage_qualified_extra_exposures": int(values[7].cpu()),
            "minimum_transition_coverage": self.minimum_transition_coverage,
            "augmentation_target_rate": self.augmentation_target_rate,
            "augmentation_slot_budget": self.augmentation_slot_budget,
            "spectrum_representation_scope": self.spectrum_representation_scope,
            "augmentation_slot_budget_remaining": max(0, self.augmentation_slot_budget - total_augmented),
            "raw_train_single_user_candidates": self.single_user_transition_candidates,
            "raw_train_coverage_qualified_candidates": self.coverage_qualified_candidates,
            "projection_dim": self.spectrum.projection_dim,
            "covariance_decay": self.spectrum.covariance_decay,
            "conflict_ema_decay": self.conflict_ema_decay,
            "conflict_rate_last": self.conflict_last_rate,
            "conflict_target_augmentation_rate_last": self.conflict_last_target_rate,
            "effective_rank_last_local": self.spectrum_effective_rank,
            "trace": str(self.spectrum_trace),
            "raw_event_pool": "literal P1 permutation of original TRAIN occurrences",
            "selection": (
                ("online spectrum probabilities" if self.event_sampling_mode == "spectrum" else "uniform raw-batch probabilities")
                + f" over transitions with coverage of at least {self.minimum_transition_coverage} distinct TRAIN users"
                + ("; eligible view slots use output-head gradient conflict" if self.conflict_sampling_mode == "conflict" else "; eligible view slots are allocated uniformly without a gradient-conflict proxy")
            ),
            "transition_coverage_source": "distinct TRAIN users sharing each adjacent full-SID transition",
            "single_user_view_exposures": 0,
            "position": "online append to current consumed raw batch",
            "uses_relation_manifest": False,
            "uses_precomputed_view_assets": False,
            "uses_probe": False,
            "uses_old_checkpoint": False,
        }
