"""Shared original P/K loader; only the approved step200 selector is replaced."""
import hashlib
import importlib
import json
from pathlib import Path

from src.data.loading.datamodules.replay_selection_datamodule import ReplaySelectionSequenceDataModule


class BaselineSelectionDataModule(ReplaySelectionSequenceDataModule):
    def __init__(self, *args, baseline_method, baseline_root, **kwargs):
        if baseline_method not in ("GradSim_Diverse", "DEALRec", "GORACS"):
            raise ValueError("Method outside approved baseline scope")
        kwargs["selection_arm"] = "random80"
        super().__init__(*args, **kwargs)
        self.baseline_method = baseline_method
        self.baseline_root = Path(baseline_root).resolve()
        self.selection_arm = "baseline_" + baseline_method
        path = self.baseline_root / "frozen_protocol.json"
        protocol = json.loads(path.read_text())
        if not protocol["execution_approved"] or protocol["max_simultaneous_gpus"] != 4:
            raise RuntimeError("Current approved scope missing")
        self.baseline_protocol_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        task = next(t for t in protocol["tasks"]
                    if t["dataset"] == self.dataset and t["method"] == baseline_method)
        if task["selected_count"] != 4*self.occurrence_count//5:
            raise RuntimeError("Frozen pool/retention mismatch")

    def _bind_selection(self, folder):
        manifest = json.loads((Path(folder) / "manifest.json").read_text())
        expected = dict(GradSim_Diverse="GradSim_Diverse_Weighted",
                        DEALRec="DEALRec", GORACS="GORACS")[self.baseline_method]
        if (manifest.get("method") != expected
                or manifest.get("baseline_protocol_sha256") != self.baseline_protocol_sha256):
            raise RuntimeError("Selection belongs to another method or scientific contract")
        super()._bind_selection(folder)

    def ensure_selection(self):
        if self.selected_positions is not None:
            return
        if self.trainer.global_step != 200 or self.consumed_batches != 25600:
            raise RuntimeError("Selector must start at committed update200")
        self._audit_prefix()  # Informational state record; no numeric parity gate.
        modules = dict(GradSim_Diverse="src.data.selection.baseline_gradsim",
                       DEALRec="src.data.selection.baseline_dealrec",
                       GORACS="src.data.selection.sports_baseline_goracs")
        selector = importlib.import_module(modules[self.baseline_method])
        folder = selector.ensure_selection(self)
        self._bind_selection(folder)

    def state_dict(self):
        state = super().state_dict()
        state.update(baseline_method=self.baseline_method,
                     baseline_protocol_sha256=self.baseline_protocol_sha256)
        return state

    def load_state_dict(self, state):
        # Preserve the parent's validation/test no-op behavior.
        from lightning.pytorch.trainer.states import TrainerFn
        fn = getattr(getattr(self.trainer, "state", None), "fn", None)
        if fn in (TrainerFn.TESTING, TrainerFn.VALIDATING, TrainerFn.PREDICTING):
            return
        if (state.get("baseline_method") != self.baseline_method
                or state.get("baseline_protocol_sha256") != self.baseline_protocol_sha256):
            raise RuntimeError("Different baseline data contract")
        super().load_state_dict(state)
