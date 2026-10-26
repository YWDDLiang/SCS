"""Online LOOM-SPICE scheduling over the original G2 copy/view actions.

Every G2 auxiliary slot keeps its original parent, target, sequence length,
position, and budget. At periodic refreshes, a small TRAIN-only sample of the
same G2 parents is evaluated under the copy and G-view actions; the resulting
low-dimensional gradient representations update one shared action policy.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightning.pytorch.trainer.states import TrainerFn

from src.data.loading.datamodules.target_matched_views_datamodule import (
    TargetMatchedViewsDataModule,
)
from src.data.loading.datamodules.relation_training_datamodule import (
    RelationTrainingSequenceDataModule,
)
from src.data.selection.loom_spice_view_scheduler import (
    LOOMSPICEViewScheduler,
    effective_rank,
)


class LOOMSPICEViewsDataModule(TargetMatchedViewsDataModule):
    def __init__(
        self,
        *args,
        loom_spice_refresh_updates=50,
        loom_spice_refresh_examples=4,
        loom_spice_seed=20261020,
        loom_spice_alpha=1.0,
        loom_spice_gamma=1.0,
        loom_spice_conflict_weight=1.0,
        loom_spice_temperature=0.7,
        loom_spice_exploration=0.05,
        loom_spice_identity_floor=0.10,
        loom_spice_score_ema=0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.target_view_variant != "views" or self.relation_arm != "u2to4_x2":
            raise ValueError("LOOM-SPICE is bound to the existing Toys G2 view assets")
        self.loom_spice_refresh_updates = int(loom_spice_refresh_updates)
        self.loom_spice_refresh_examples = int(loom_spice_refresh_examples)
        self.loom_spice_seed = int(loom_spice_seed)
        if self.loom_spice_refresh_updates < 1 or self.loom_spice_refresh_examples < 2:
            raise ValueError("refresh interval must be positive and at least two probe parents are required")
        self.loom_spice = LOOMSPICEViewScheduler(
            2,
            alpha=float(loom_spice_alpha),
            gamma=float(loom_spice_gamma),
            conflict_weight=float(loom_spice_conflict_weight),
            temperature=float(loom_spice_temperature),
            exploration=float(loom_spice_exploration),
            min_identity_probability=float(loom_spice_identity_floor),
            score_ema_decay=float(loom_spice_score_ema),
        )
        self.loom_policy = torch.tensor([0.5, 0.5], dtype=torch.float32)
        rank = int(os.environ.get("RANK", "0"))
        self.loom_rank = rank
        self.loom_action_generator = torch.Generator(device="cpu").manual_seed(
            self.loom_spice_seed + rank
        )
        self.loom_last_refresh_update = -1
        self.loom_refresh_count = 0
        self.loom_action_counts = np.zeros(2, dtype=np.int64)
        self.loom_probe_seconds = 0.0
        self.loom_trace = self.relation_output / f"loom_spice_rank{rank}.jsonl"
        if rank == 0:
            self.loom_trace.parent.mkdir(parents=True, exist_ok=True)
            self.loom_trace.write_text("")

    @staticmethod
    def _move_batch(model_input, label_data, device):
        model_input.mask = model_input.mask.to(device)
        model_input.transformed_sequences = {
            key: value.to(device)
            for key, value in model_input.transformed_sequences.items()
        }
        label_data.labels = {
            key: value.to(device) for key, value in label_data.labels.items()
        }
        label_data.label_location = {
            key: value.to(device)
            for key, value in label_data.label_location.items()
        }
        label_data.attention_mask = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in label_data.attention_mask.items()
        }
        return model_input, label_data

    @staticmethod
    def _proxy(model, batch):
        model_input, label_data = batch
        hierarchy = model.num_hierarchies
        targets = label_data.labels["sequence_data"].reshape(
            model_input.mask.shape[0], hierarchy
        ).long()
        inputs = {
            model.feature_to_model_input_map.get(name, name): value
            for name, value in model_input.transformed_sequences.items()
        }
        with torch.enable_grad():
            output = model(
                attention_mask_encoder=model_input.mask,
                future_ids=targets,
                **inputs,
            )[:, :-1]
            losses = torch.stack(
                [
                    F.cross_entropy(
                        model.decoder.decoder_mlp[level](output[:, level]),
                        targets[:, level],
                        reduction="none",
                    )
                    for level in range(hierarchy)
                ],
                dim=1,
            ).sum(dim=1)
            output_gradient = torch.autograd.grad(losses.sum(), output)[0]
        return (output.detach() * output_gradient.detach()).flatten(1).float()

    def _probe_details(self, ordinals, changed):
        ids = np.asarray(self.copy_ids[ordinals], dtype=np.int64)
        rows = np.asarray(self.occurrences[ids], dtype=np.int32).copy()
        keys = np.asarray(self.occurrence_to_key[ids], dtype=np.int64)
        return dict(
            rows=rows,
            ids=ids,
            auxiliary=np.ones(len(ids), dtype=bool),
            ordinal=np.asarray(ordinals, dtype=np.int64),
            changed=np.full(len(ids), bool(changed), dtype=bool),
            novel=(np.asarray(self.view_novel[ordinals], dtype=bool) if changed else np.zeros(len(ids), dtype=bool)),
            flags=np.zeros(len(ids), dtype=np.uint8),
            keys=keys,
            parent_keys=keys,
        )

    def _sample_probe_ordinals(self, update):
        generator = np.random.default_rng(
            self.loom_spice_seed + 1009 * int(update) + 37 * self.loom_rank
        )
        selected = []
        seen = set()
        while len(selected) < self.loom_spice_refresh_examples:
            candidates = generator.integers(
                0, len(self.copy_ids), size=self.loom_spice_refresh_examples * 4
            )
            for ordinal in candidates.tolist():
                if ordinal not in seen and bool(self.view_changed[ordinal]):
                    selected.append(int(ordinal))
                    seen.add(int(ordinal))
                    if len(selected) == self.loom_spice_refresh_examples:
                        break
        return np.asarray(selected, dtype=np.int64)

    def _refresh_policy(self, model, update):
        if update == self.loom_last_refresh_update:
            return
        if update < 200 or (update - 200) % self.loom_spice_refresh_updates:
            return
        ordinals = self._sample_probe_ordinals(update)
        copy_batch = self._collate_view_batch(self._probe_details(ordinals, False))
        view_batch = self._collate_view_batch(self._probe_details(ordinals, True))
        device = next(model.parameters()).device
        copy_batch = self._move_batch(*copy_batch, device)
        view_batch = self._move_batch(*view_batch, device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        was_training = model.training
        model.eval()
        try:
            with torch.enable_grad():
                copy_proxy = self._proxy(model, copy_batch)
                view_proxy = self._proxy(model, view_batch)
        finally:
            model.train(was_training)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        self.loom_probe_seconds += elapsed

        local = torch.stack((copy_proxy, view_proxy), dim=1).reshape(
            -1, copy_proxy.shape[1]
        ).detach().contiguous()
        if dist.is_available() and dist.is_initialized():
            gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local)
            representations = torch.cat(gathered, dim=0)
            world = dist.get_world_size()
        else:
            representations = local
            world = 1
        groups = torch.arange(
            world * self.loom_spice_refresh_examples, device=device
        ).repeat_interleave(2)
        actions = torch.tensor([0, 1], device=device).repeat(world * self.loom_spice_refresh_examples)
        refresh = self.loom_spice.refresh(
            representations,
            groups,
            actions,
            step=int(update),
        )
        unique_groups = torch.arange(len(groups) // 2, device=device)
        group_policy = self.loom_spice.action_probabilities(unique_groups)
        self.loom_policy = group_policy.mean(dim=0).detach().float().cpu()

        # Fold the selected probe representations into the running Fisher and
        # direction state used at the next refresh. This is a tiny TRAIN-only
        # probe; no validation or test inputs enter the scheduler.
        generator = torch.Generator(device="cpu").manual_seed(
            self.loom_spice_seed ^ (int(update) * 7919)
        )
        selected = torch.multinomial(
            group_policy.detach().float().cpu(), 1, generator=generator
        ).squeeze(1)
        chosen_rows = torch.arange(len(selected), device=device) * 2 + selected.to(device)
        self.loom_spice.observe(representations.index_select(0, chosen_rows))
        self.loom_last_refresh_update = int(update)
        self.loom_refresh_count += 1
        if self.loom_rank == 0:
            record = dict(
                update=int(update),
                probe_parents=int(len(selected)),
                policy=self.loom_policy.tolist(),
                mean_loom_alignment=float(refresh.loom_alignment.mean().cpu()),
                mean_spectral_novelty=float(refresh.novelty.mean().cpu()),
                mean_gradient_conflict=float(refresh.conflict.mean().cpu()),
                mean_combined_score=float(refresh.combined_score.mean().cpu()),
                effective_rank=float(effective_rank(self.loom_spice.covariance).cpu()),
                probe_seconds=elapsed,
            )
            with self.loom_trace.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

    def _apply_policy(self, details):
        auxiliary_rows = np.flatnonzero(details["auxiliary"])
        if not len(auxiliary_rows):
            return
        policy = self.loom_policy.clamp_min(0)
        policy /= policy.sum()
        choices = torch.multinomial(
            policy.expand(len(auxiliary_rows), -1),
            1,
            generator=self.loom_action_generator,
        ).squeeze(1).numpy().astype(bool)
        ordinals = details["ordinal"]
        details["changed"][auxiliary_rows] = choices & self.view_changed[ordinals]
        details["novel"][auxiliary_rows] = choices & self.view_novel[ordinals]
        self.loom_action_counts += np.bincount(
            choices.astype(np.int64), minlength=2
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.training:
            return batch

        def consume(node):
            if isinstance(node, dict) and set(node) == {"relation_batch_index"}:
                index = int(node["relation_batch_index"])
                rank = int(self.trainer.global_rank)
                if index % 4 != rank or index - rank != self.consumed_batches:
                    raise RuntimeError("LOOM-SPICE rank cursor mismatch")
                if index >= self.boundary:
                    self._activate()
                    self._refresh_policy(
                        self.trainer.lightning_module,
                        int(self.trainer.global_step),
                    )
                details = self.batch_details_at(index)
                if index >= self.boundary:
                    self._apply_policy(details)
                self.last_occurrence_indices = details["ids"].copy()
                output = self._collate_view_batch(details)
                exposure = self.exposure
                exposure["batches"] += 1
                exposure["samples"] += len(details["rows"])
                exposure["auxiliary_slots"] += int(details["auxiliary"].sum())
                exposure["copies"] += int((details["auxiliary"] & ~details["changed"]).sum())
                exposure["context_changes"] += int(details["changed"].sum())
                exposure["generated_changed"] += int(details["changed"].sum())
                exposure["generated_novel_in_source"] += int(details["novel"].sum())
                support = self.key_support[details["keys"]]
                for name, mask in (
                    ("support1", support == 1),
                    ("support2to4", (support >= 2) & (support <= 4)),
                    ("support5plus", support >= 5),
                ):
                    count = int(mask.sum())
                    exposure[name] += count
                    exposure[name + "_weight"] += count / len(details["rows"])
                return super(RelationTrainingSequenceDataModule, self).on_before_batch_transfer(
                    (*output, {"ddp_global_batch_index": index}), dataloader_idx
                )
            if not isinstance(node, (tuple, list)):
                return node
            children = [consume(child) for child in node]
            return tuple(children) if isinstance(node, tuple) else children

        return consume(batch)

    def state_dict(self):
        state = super().state_dict()
        state.update(
            loom_spice_format_version=1,
            loom_spice_scheduler=self.loom_spice.state_dict(),
            loom_spice_policy=self.loom_policy.clone(),
            loom_spice_last_refresh_update=self.loom_last_refresh_update,
            loom_spice_refresh_count=self.loom_refresh_count,
            loom_spice_action_counts=self.loom_action_counts.copy(),
            loom_spice_probe_seconds=self.loom_probe_seconds,
            loom_spice_action_rng=self.loom_action_generator.get_state(),
        )
        return state

    def method_summary(self):
        device = self.trainer.strategy.root_device
        counts = torch.as_tensor(self.loom_action_counts, dtype=torch.int64, device=device)
        seconds = torch.tensor(self.loom_probe_seconds, dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(seconds, op=dist.ReduceOp.SUM)
        return dict(
            method="LOOM-SPICE over the original G2 copy/view actions",
            action_names=["copy", "G2 view"],
            action_counts=counts.cpu().tolist(),
            final_action_probabilities=self.loom_policy.tolist(),
            refresh_updates=self.loom_spice_refresh_updates,
            refresh_examples_per_rank=self.loom_spice_refresh_examples,
            refresh_count=self.loom_refresh_count,
            probe_seconds_summed_over_ranks=float(seconds.cpu()),
            effective_rank=(
                float(effective_rank(self.loom_spice.covariance).cpu())
                if self.loom_spice.covariance is not None
                else 0.0
            ),
            last_refresh_update=self.loom_last_refresh_update,
            trace=str(self.loom_trace),
        )

    def load_state_dict(self, state):
        if getattr(getattr(self.trainer, "state", None), "fn", None) in (
            TrainerFn.TESTING,
            TrainerFn.VALIDATING,
            TrainerFn.PREDICTING,
        ):
            return
        super().load_state_dict(state)
        if state.get("loom_spice_format_version") != 1:
            raise RuntimeError("checkpoint lacks the LOOM-SPICE scheduler state")
        self.loom_spice.load_state_dict(state["loom_spice_scheduler"])
        self.loom_policy = torch.as_tensor(state["loom_spice_policy"]).cpu().float()
        self.loom_last_refresh_update = int(state["loom_spice_last_refresh_update"])
        self.loom_refresh_count = int(state["loom_spice_refresh_count"])
        self.loom_action_counts = np.asarray(state["loom_spice_action_counts"], dtype=np.int64)
        self.loom_probe_seconds = float(state["loom_spice_probe_seconds"])
        self.loom_action_generator.set_state(state["loom_spice_action_rng"])
