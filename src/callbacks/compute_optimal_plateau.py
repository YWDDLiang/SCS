from __future__ import annotations

import json
from pathlib import Path

from lightning import Callback, LightningModule, Trainer

from src.data.selection.compute_scheduler import (
    ValidationPoint,
    decide_schedule_action,
)


class ComputeOptimalPlateau(Callback):
    """Stop a training stage when validation gain justifies data refinement."""

    def __init__(
        self,
        decision_path: str,
        *,
        monitor: str = "val/ndcg@10",
        total_update_budget: int,
        refinements_used: int = 0,
        max_refinements: int = 2,
        window: int = 5,
        min_step: int = 400,
        trigger_gain_ratio: float = 0.10,
        scoring_cost_updates: int = 15,
        min_updates_after_refinement: int = 50,
        min_replacement_fraction: float = 0.05,
        max_replacement_fraction: float = 0.15,
        refinement_started_step: int | None = None,
        pre_refinement_best: float | None = None,
        probe_updates: int = 20,
        min_relative_acceptance_gain: float = 0.002,
    ) -> None:
        self.decision_path = Path(decision_path)
        self.monitor = monitor
        self.total_update_budget = total_update_budget
        self.refinements_used = refinements_used
        self.max_refinements = max_refinements
        self.window = window
        self.min_step = min_step
        self.trigger_gain_ratio = trigger_gain_ratio
        self.scoring_cost_updates = scoring_cost_updates
        self.min_updates_after_refinement = min_updates_after_refinement
        self.min_replacement_fraction = min_replacement_fraction
        self.max_replacement_fraction = max_replacement_fraction
        self.refinement_started_step = refinement_started_step
        self.pre_refinement_best = pre_refinement_best
        self.probe_updates = probe_updates
        self.min_relative_acceptance_gain = min_relative_acceptance_gain
        self.points: list[ValidationPoint] = []
        self.last_evaluated_step = -1

    def on_validation_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking or trainer.global_step == self.last_evaluated_step:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            raise RuntimeError(f"missing monitored metric: {self.monitor}")
        point = ValidationPoint(trainer.global_step, float(metric.detach().cpu()))
        self.points.append(point)
        self.last_evaluated_step = trainer.global_step
        decision = decide_schedule_action(
            self.points,
            current_step=trainer.global_step,
            total_update_budget=self.total_update_budget,
            refinements_used=self.refinements_used,
            max_refinements=self.max_refinements,
            window=self.window,
            min_step=self.min_step,
            trigger_gain_ratio=self.trigger_gain_ratio,
            scoring_cost_updates=self.scoring_cost_updates,
            min_updates_after_refinement=self.min_updates_after_refinement,
            min_replacement_fraction=self.min_replacement_fraction,
            max_replacement_fraction=self.max_replacement_fraction,
            refinement_started_step=self.refinement_started_step,
            pre_refinement_best=self.pre_refinement_best,
            probe_updates=self.probe_updates,
            min_relative_acceptance_gain=self.min_relative_acceptance_gain,
        )
        if trainer.is_global_zero:
            self.decision_path.parent.mkdir(parents=True, exist_ok=True)
            payload = decision.to_dict()
            payload["monitor"] = self.monitor
            payload["refinement_started_step"] = self.refinement_started_step
            payload["pre_refinement_best"] = self.pre_refinement_best
            payload["probe_updates"] = self.probe_updates
            payload["min_relative_acceptance_gain"] = (
                self.min_relative_acceptance_gain
            )
            payload["post_refinement_best"] = (
                max(item.value for item in self.points) if self.points else None
            )
            payload["validation_points"] = [
                {"step": item.step, "value": item.value} for item in self.points
            ]
            self.decision_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        if decision.action != "train":
            trainer.should_stop = True

    def state_dict(self) -> dict[str, object]:
        return {
            "points": [(point.step, point.value) for point in self.points],
            "last_evaluated_step": self.last_evaluated_step,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.points = [
            ValidationPoint(int(step), float(value))
            for step, value in state_dict.get("points", [])
        ]
        self.last_evaluated_step = int(state_dict.get("last_evaluated_step", -1))
