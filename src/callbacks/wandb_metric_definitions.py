from __future__ import annotations

from lightning import Callback, Trainer
from lightning.pytorch.loggers import WandbLogger


class WandbMetricDefinitions(Callback):
    """Add useful W&B run summaries without changing metric computation."""

    METRIC_SUMMARIES = {
        "train/loss_step": "min",
        "train/loss_epoch": "min",
        "val/loss": "min",
        "val/ndcg@5": "max",
        "val/ndcg@10": "max",
        "val/recall@5": "max",
        "val/recall@10": "max",
    }

    def on_fit_start(self, trainer: Trainer, pl_module) -> None:
        for logger in trainer.loggers:
            if not isinstance(logger, WandbLogger):
                continue
            experiment = logger.experiment
            for metric_name, summary in self.METRIC_SUMMARIES.items():
                experiment.define_metric(metric_name, summary=summary)
