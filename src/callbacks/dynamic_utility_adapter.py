"""Scorer bridge: original candidate/reference batches, no DDP gradient averaging."""
class DynamicUtilityAdapter:
    def __init__(self,candidate_batch_size=512,reference_batch_size=8):
        self.candidate_batch_size=candidate_batch_size
        self.reference_batch_size=reference_batch_size

    def score(self,*,model,optimizer,datamodule,global_step,previous_state):
        from src.data.selection.dynamic_utility import score_dynamic_utility
        return score_dynamic_utility(model,optimizer,
            datamodule.scoring_batches(batch_size=self.candidate_batch_size),
            datamodule.reference_batches(batch_size=self.reference_batch_size),
            len(datamodule.group_ids),group_ids=datamodule.group_ids,step=global_step,
            previous_losses=None if previous_state is None else previous_state['losses'],
            reference_count=1024,provenance={**datamodule.assets['metadata'],
                **getattr(model,'dynamic_scoring_provenance',{}),
                'current_optimizer_step':global_step,'asset_manifest':datamodule.manifest})
