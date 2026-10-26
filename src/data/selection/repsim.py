from __future__ import annotations

import torch
import torch.nn.functional as F


def normalized_first_prediction_representations(hidden_states: torch.Tensor) -> torch.Tensor:
    """Return one L2-normalized target-free decoder prediction vector per example.

    TIGER returns decoder states shaped ``[batch, hierarchy, hidden]`` and uses
    the first state to predict the first Semantic-ID level. Under causal decoder
    masking this state is conditioned on the candidate history and decoder start
    token, but not on an observed target Semantic-ID level.
    """
    hidden = torch.as_tensor(hidden_states)
    if hidden.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, levels, hidden]")
    if hidden.shape[0] == 0 or hidden.shape[1] == 0 or hidden.shape[2] == 0:
        raise ValueError("hidden_states dimensions must be non-empty")
    return F.normalize(hidden[:, 0, :].float(), p=2, dim=-1)


def chunked_max_reference_cosine(
    candidate_representations: torch.Tensor,
    reference_representations: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Compute max cosine-to-reference without materializing an NxR matrix."""
    candidates = torch.as_tensor(candidate_representations)
    references = torch.as_tensor(reference_representations)
    if candidates.ndim != 2 or references.ndim != 2:
        raise ValueError("candidate and reference representations must be rank two")
    if candidates.shape[1] != references.shape[1]:
        raise ValueError("candidate and reference representation dimensions must match")
    if candidates.shape[0] == 0 or references.shape[0] == 0:
        raise ValueError("candidate and reference representations must be non-empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    references = F.normalize(references.float(), p=2, dim=-1)
    scores = torch.empty(candidates.shape[0], dtype=torch.float32)
    for start in range(0, candidates.shape[0], chunk_size):
        end = min(start + chunk_size, candidates.shape[0])
        chunk = F.normalize(candidates[start:end].float(), p=2, dim=-1)
        scores[start:end] = (chunk @ references.T).amax(dim=1).cpu()
    return scores
