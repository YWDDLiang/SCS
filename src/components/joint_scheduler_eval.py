from __future__ import annotations

import torch
from torchmetrics.utilities.distributed import gather_all_tensors

from src.components.eval_metrics import SIDRetrievalEvaluator


class RecordingSIDRetrievalEvaluator(SIDRetrievalEvaluator):
    """SID evaluator that retains validation user IDs and per-user NDCG@10."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._recorded_user_ids: list[torch.Tensor] = []
        self._recorded_ndcg10: list[torch.Tensor] = []

    def reset(self) -> None:
        super().reset()
        self._recorded_user_ids = []
        self._recorded_ndcg10 = []

    def __call__(
        self,
        marginal_probs: torch.Tensor,
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        user_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__call__(
            marginal_probs=marginal_probs,
            generated_ids=generated_ids,
            labels=labels,
            **kwargs,
        )
        if user_ids is None:
            raise RuntimeError(
                "RecordingSIDRetrievalEvaluator requires validation user IDs"
            )
        batch_size, num_candidates, num_hierarchies = generated_ids.shape
        flat_user_ids = torch.as_tensor(user_ids, device=generated_ids.device).flatten()
        if flat_user_ids.numel() != batch_size:
            raise ValueError("user_ids must contain one value per validation example")

        reshaped_labels = labels.reshape(batch_size, 1, num_hierarchies)
        target = torch.all(generated_ids == reshaped_labels, dim=2)
        top_k = min(10, num_candidates)
        ranked = torch.topk(marginal_probs, top_k, dim=1).indices
        ranked_relevance = target.gather(1, ranked)
        discounts = torch.log2(
            torch.arange(
                2,
                top_k + 2,
                device=marginal_probs.device,
                dtype=marginal_probs.dtype,
            )
        )
        ndcg10 = (ranked_relevance.to(marginal_probs.dtype) / discounts).sum(dim=1)
        self._recorded_user_ids.append(flat_user_ids.detach())
        self._recorded_ndcg10.append(ndcg10.detach())

    def gathered_per_user_ndcg10(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._recorded_user_ids:
            raise RuntimeError("no per-user validation observations were recorded")
        user_ids = torch.cat(self._recorded_user_ids)
        values = torch.cat(self._recorded_ndcg10)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            user_ids = torch.cat(gather_all_tensors(user_ids))
            values = torch.cat(gather_all_tensors(values))
        order = torch.argsort(user_ids, stable=True)
        user_ids = user_ids[order].cpu()
        values = values[order].cpu()
        if torch.unique(user_ids).numel() != user_ids.numel():
            raise RuntimeError("validation user IDs must be unique for paired comparison")
        return user_ids, values
