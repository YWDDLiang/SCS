"""Optional eval-only elimination of the duplicate TIGER encoder invocation."""
import torch
from src.models.modules.semantic_id.tiger_generation_model import SemanticIDEncoderDecoder


class EfficientSemanticIDEncoderDecoder(SemanticIDEncoderDecoder):
    """Reuse encoder output only inside one deterministic, no-grad eval_step.

    Training and standalone generate/model_step remain the original implementation.
    The cache is a plain transient dict, never a parameter/buffer/state_dict key.
    """
    @staticmethod
    def _tensor_key(value):
        if value is None:
            return None
        try:
            version = value._version
        except RuntimeError:  # inference tensors have no version counter
            version = None
        return (id(value), value.data_ptr(), version, tuple(value.shape),
                tuple(value.stride()), value.storage_offset(), value.dtype, value.device)

    def encoder_forward_pass(self, attention_mask, input_ids, user_id):
        cache = getattr(self, '_eval_encoder_cache', None)
        if cache is None:
            return super().encoder_forward_pass(attention_mask, input_ids, user_id)
        key = tuple(self._tensor_key(value) for value in (attention_mask, input_ids, user_id))
        if cache['key'] is not None:
            if key != cache['key']:
                raise RuntimeError('Encoder inputs changed inside the eval-only cache scope')
            cache['hits'] += 1
            return cache['output']
        cache['key'] = key
        cache['output'] = super().encoder_forward_pass(attention_mask, input_ids, user_id)
        cache['computations'] += 1
        return cache['output']

    def eval_step(self, batch, loss_to_aggregate):
        # Each T5 attention also uses self.training for functional dropout.
        # Checking every submodule covers nn.Dropout and functional dropout flags.
        eligible = (not torch.is_grad_enabled() and
                    not any(module.training for module in self.modules()) and
                    not any(getattr(module, 'is_training', False) for module in self.modules()))
        if not eligible:
            return super().eval_step(batch, loss_to_aggregate)
        if getattr(self, '_eval_encoder_cache', None) is not None:
            raise RuntimeError('Nested eval_step encoder cache scopes are unsupported')
        cache = dict(key=None, output=None, computations=0, hits=0)
        self._eval_encoder_cache = cache
        try:
            return super().eval_step(batch, loss_to_aggregate)
        finally:
            self._last_eval_encoder_computations = cache['computations']
            self._last_eval_encoder_cache_hits = cache['hits']
            self._eval_encoder_cache = None
