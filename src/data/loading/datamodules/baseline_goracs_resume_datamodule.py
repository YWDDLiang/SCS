"""Resume a saved step200 model only after complete actual ITRA publication."""
import json,hashlib
from pathlib import Path
from src.data.loading.datamodules.baseline_selection_datamodule import BaselineSelectionDataModule as Original
class BaselineSelectionDataModule(Original):
    def ensure_selection(self):
        if self.selected_positions is not None:
            return
        if self.baseline_method != 'GORACS' or self.trainer.global_step != 200 or self.consumed_batches != 25600:
            raise RuntimeError('Resume requires original committed GORACS step200')
        folder=Path(self.selection_output_dir)
        result=json.loads((folder/'cpu_result.json').read_text())
        manifest=json.loads((folder/'manifest.json').read_text())
        if (result.get('status')!='completed_actual_full_pool_ITRA'
                or result.get('actual_full_pool_ITRA_completed') is not True
                or result.get('original_occurrences')!=self.occurrence_count
                or result.get('selected_occurrences')!=4*self.occurrence_count//5
                or manifest.get('CPU_ITRA_result_sha256')!=hashlib.sha256((folder/'cpu_result.json').read_bytes()).hexdigest()):
            raise RuntimeError('A greedy-only or mismatched selection cannot resume training')
        self._bind_selection(folder)
