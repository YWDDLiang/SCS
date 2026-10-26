"""Actual TIGER gradients for the approved baseline selectors.

Linear-weight gradients are stored as exact activation/backprop factors, not as
output-head proxies or random projections. Scoring uses individual TRAIN queries.
"""
from contextlib import contextmanager
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.data.selection.dynamic_utility import actual_heads


@contextmanager
def scoring_state(model, distributed_model=None):
    device = next(model.parameters()).device
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
    modes = [(m, m.training, getattr(m, "is_training", None)) for m in model.modules()]
    sync = getattr(distributed_model, "require_backward_grad_sync", None)
    forward_sync = getattr(distributed_model, "require_forward_param_sync", None)
    try:
        model.eval()
        for m, _, special in modes:
            if special is not None:
                m.is_training = False
        if sync is not None:
            distributed_model.require_backward_grad_sync = False
        if forward_sync is not None:
            distributed_model.require_forward_param_sync = False
        with torch.inference_mode(False), torch.enable_grad():
            yield
    finally:
        for m, training, special in modes:
            m.training = training
            if special is not None:
                m.is_training = special
        if sync is not None:
            distributed_model.require_backward_grad_sync = sync
        if forward_sync is not None:
            distributed_model.require_forward_param_sync = forward_sync
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state(rng[3], device)


def final_attention_projections(model):
    layers = []
    for projection in ("q", "k", "v", "o"):
        suffix = "decoder.block.3.layer.0.SelfAttention." + projection
        matches = [(n, m) for n, m in model.named_modules() if n.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError("Expected one final decoder projection: " + projection)
        name, layer = matches[0]
        if not isinstance(layer, torch.nn.Linear) or layer.bias is not None:
            raise RuntimeError("Projection gradient mapping changed")
        layers.append((name, layer))
    return layers


def query_loss(model, batch):
    data, labels = batch
    device = next(model.parameters()).device
    mask = data.mask.to(device)
    targets = next(iter(labels.labels.values())).reshape(mask.shape[0], 4).to(device).long()
    if len(targets) != 1:
        raise ValueError("Per-query gradient call must contain one original training query")
    fields = {model.feature_to_model_input_map.get(k, k): v.to(device)
              for k, v in data.transformed_sequences.items()}
    hidden = model.forward(attention_mask_encoder=mask, future_ids=targets, **fields)[:, :-1]
    heads = actual_heads(model)
    if len(heads) != 4 or hidden.shape[1] != 4 or torch.any(targets < 0):
        raise ValueError("Four valid SID target heads required")
    logits = torch.stack([head(hidden[:, i]) for i, head in enumerate(heads)], dim=1)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="mean")


def query_gradient_features(model, batch, need_factors=True, need_full_norm=True,
                            verify_dense=False):
    projections = final_attention_projections(model) if need_factors else []
    captured = {}
    handles = []
    for name, layer in projections:
        def hook(module, inputs, output, key=name):
            if key in captured:
                raise RuntimeError("Projection invoked twice; review gradient checkpointing")
            captured[key] = (inputs[0].detach(), output)
        handles.append(layer.register_forward_hook(hook))
    try:
        loss = query_loss(model, batch)
        params = [p for p in model.parameters() if p.requires_grad] if need_full_norm else []
        if verify_dense and not need_full_norm:
            params = [layer.weight for _, layer in projections]
        outputs = [captured[name][1] for name, _ in projections]
        grads = torch.autograd.grad(loss, params+outputs, allow_unused=True)
        param_grads = grads[:len(params)]
        output_grads = grads[len(params):]
        if need_full_norm:
            full_norm = torch.sqrt(sum(g.detach().double().square().sum()
                                       for g in param_grads if g is not None))
        else:
            full_norm = None
        factors = []
        attention_squared = loss.new_zeros((), dtype=torch.float64)
        errors = []
        for (name, layer), gradient in zip(projections, output_grads):
            if gradient is None:
                raise RuntimeError("Disconnected last attention projection")
            a = captured[name][0][0].float().detach()
            b = gradient[0].float().detach()
            if a.ndim != 2 or a.shape[0] != b.shape[0]:
                raise RuntimeError("Unexpected projection sequence dimensions")
            # dW = B^T A; compute its Frobenius norm without storing a bank of dW.
            aa = a.double() @ a.double().T
            bb = b.double() @ b.double().T
            attention_squared += (aa*bb).sum()
            factors.append((a, b))
            if verify_dense:
                dense = param_grads[next(i for i, p in enumerate(params) if p is layer.weight)]
                rebuilt = b.T @ a
                errors.append(float((dense-rebuilt).abs().max().item()))
        attention_norm = attention_squared.clamp_min(0).sqrt() if need_factors else None
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite selector query loss")
        if full_norm is not None and not torch.isfinite(full_norm):
            raise RuntimeError("Non-finite full-parameter gradient")
        return dict(loss=float(loss.detach()), factors=factors,
                    attention_norm=attention_norm, full_norm=full_norm,
                    dense_reconstruction_max_errors=errors)
    finally:
        for handle in handles:
            handle.remove()


def factor_inner_product(left, right):
    """Exact parameter-space dot product, including cross-token terms."""
    return sum(((a.double() @ c.double().T) * (b.double() @ d.double().T)).sum()
               for (a, b), (c, d) in zip(left, right))


def factor_reference_matrix(features):
    """Mean of individually normalized gradients; zero norms contribute zero."""
    count = len(features)
    if count == 0:
        raise ValueError("Empty reference")
    result = [torch.zeros((b.shape[1], a.shape[1]), device=a.device, dtype=torch.float64)
              for a, b in features[0]["factors"]]
    for f in features:
        norm = float(f["attention_norm"])
        if norm > 0:
            for i, (a, b) in enumerate(f["factors"]):
                result[i] += (b.double().T @ a.double()) / (norm*count)
    return result


def factor_alignment(factors, dense_reference, norm):
    if float(norm) == 0:
        return 0.
    value = sum(((a.double() @ ref.T) * b.double()).sum()
                for (a, b), ref in zip(factors, dense_reference))
    return float(value/norm)
