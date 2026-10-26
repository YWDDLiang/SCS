from __future__ import annotations

from pathlib import Path

from lightning import Callback, LightningModule, Trainer
import torch

from src.components.joint_scheduler_eval import RecordingSIDRetrievalEvaluator


class PerUserValidationWriter(Callback):
    """Write paired per-user observations without repeating inference.

    Fires on both the validation and the test evaluation, so the same per-user
    artifact can be produced for either split. The output directory is chosen by
    the caller, so the split is recorded by the path.
    """

    def __init__(self, output_dir: str, step_override: int | None = None) -> None:
        self.output_dir = Path(output_dir)
        if step_override is not None and step_override < 0:
            raise ValueError("step_override must be non-negative")
        self.step_override = step_override
        self.last_written_step = -1

    def on_validation_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._write(trainer, pl_module)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._write(trainer, pl_module)

    def _write(self, trainer: Trainer, pl_module: LightningModule) -> None:
        observation_step = (
            trainer.global_step
            if self.step_override is None
            else self.step_override
        )
        if trainer.sanity_checking or observation_step == self.last_written_step:
            return
        evaluator = pl_module.evaluator
        if not isinstance(evaluator, RecordingSIDRetrievalEvaluator):
            raise RuntimeError(
                "PerUserValidationWriter requires RecordingSIDRetrievalEvaluator"
            )
        user_ids, values = evaluator.gathered_per_user_ndcg10()
        self.last_written_step = observation_step
        if not trainer.is_global_zero:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / f"step_{observation_step:08d}.pt"
        torch.save(
            {
                "format_version": 1,
                "global_step": observation_step,
                "metric": "ndcg@10",
                "user_ids": user_ids,
                "values": values,
                "mean": float(values.double().mean()),
            },
            output,
        )

    def state_dict(self) -> dict[str, int]:
        return {"last_written_step": self.last_written_step}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.last_written_step = int(state_dict.get("last_written_step", -1))
