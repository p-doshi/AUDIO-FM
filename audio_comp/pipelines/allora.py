"""ALLoRA (Huang & Balestriero, arXiv:2410.09692) -- an adaptive-gradient
alternative to standard LoRA, adopted after a research pass comparing 7
candidate LoRA alternatives the user proposed (see journal.md 2026-08-17).
Chosen over MoRA/RandLoRA/Delta-LoRA/KRAdapter/GraLoRA/"Why LoRA Fails to
Forget" because those five diagnose LoRA as having *too little* effective
capacity (motivated by memorization/large-adaptation-budget/backdoor-
removal problems) -- this project's own evidence points the other way
(LoRA underperforming a frozen linear probe, 1-epoch beating 10-epoch on
the same fold), which reads as an overfitting/optimization failure, not
an expressivity ceiling. ALLoRA's own paper targets exactly this regime
(limited data, short fine-tuning runs) and its fix requires zero change
to per-model target_modules/architecture -- only a different gradient
rule -- making it the lowest-implementation-risk option across a
14-model heterogeneous roster.

**Mechanism, read directly from the paper's Definition/section 4.2 (not
guessed from the abstract)**: standard LoRA computes f(x) = W + eta*BA*x
with eta=alpha/r. ALLoRA removes eta from the forward pass entirely
(f-tilde = f, no output scaling) and instead rescales the GRADIENT before
it's used to compute d(loss)/dA and d(loss)/dB (never touching dropout,
removed entirely; ALLoRA is dropout-free). Per output-row i of the
combined update BA in R^(n1 x n2):
    g_tilde_i = g_i / sqrt(||(BA)_{i,:}|| + 1/eta^2),   i in [n1]
where g_i is the standard gradient flowing into row i's contribution.
**Critical detail from the paper's own "one more implementation detail"
paragraph**: this rescaling applies ONLY to the gradients feeding A and B
-- the gradient propagated back to the layer's *input* (needed for
earlier layers) is left untouched. This is why a plain
`tensor.register_hook()` isn't sufficient (a hook rewrites grad_output
uniformly for everything downstream) -- a custom autograd.Function is
used instead, computing two different quantities from the same
grad_output: an unscaled one for grad_input, a rescaled one for grad_A/
grad_B.

Because g_i = delta_i * x (chain rule: the i-th output channel's
contribution to (BA)_{i,:}'s gradient is exactly the i-th component of
the output gradient, delta_i, times the input x), rescaling g row-wise by
s_i is mathematically identical to rescaling delta (the output-gradient
vector, shape [n1], the natural tensor the module's backward() already
receives) by the same s_i BEFORE it's used to compute grad_A/grad_B. This
is what `_ALLoRAFunction.backward()` below actually does -- it never
needs to materialize the full BA gradient matrix, only the row norms of
BA itself (which is a cheap, small r-rank matrix, r=8 here).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ALLoRAFunction(torch.autograd.Function):
    """out = x @ A^T @ B^T (matches peft's own lora_A/lora_B nn.Linear
    weight-shape convention: A.weight is [r, in_features], B.weight is
    [out_features, r]). No dropout, no alpha/r scaling -- both removed
    per ALLoRA's own design (see module docstring)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, eta: float):
        ctx.save_for_backward(x, A, B)
        ctx.eta = eta
        # x: [..., in_features], A: [r, in_features], B: [out_features, r]
        return x @ A.T @ B.T  # [..., out_features]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, A, B, = ctx.saved_tensors
        eta = ctx.eta

        # Row norms of BA (row i = output channel i's combined low-rank
        # update), computed fresh each backward call from the CURRENT A/B
        # -- cheap: B [out,r] @ A [r,in] is a rank-r matrix, r=8 here.
        BA = B @ A  # [out_features, in_features]
        row_norms = BA.norm(dim=1)  # [out_features]
        scale = 1.0 / torch.sqrt(row_norms + 1.0 / (eta ** 2))  # [out_features]

        # grad_input uses the ORIGINAL (unscaled) grad_output -- per the
        # paper's explicit instruction not to modify the gradient
        # propagated to earlier layers.
        grad_input = grad_output @ B @ A if ctx.needs_input_grad[0] else None

        # grad_A / grad_B use the RESCALED grad_output (per-output-channel
        # scale, broadcasts over all leading/batch dims).
        scaled_grad_output = grad_output * scale  # broadcasts [..., out] * [out]

        grad_A = grad_B = None
        if ctx.needs_input_grad[1]:
            # d(loss)/dA = B^T @ scaled_grad_output^T @ x, contracted over
            # the batch/leading dims -- flatten to 2D first for a plain matmul.
            flat_sgo = scaled_grad_output.reshape(-1, scaled_grad_output.shape[-1])  # [N, out]
            flat_x = x.reshape(-1, x.shape[-1])  # [N, in]
            grad_A = (B.T @ flat_sgo.T) @ flat_x  # [r, in]
        if ctx.needs_input_grad[2]:
            flat_sgo = scaled_grad_output.reshape(-1, scaled_grad_output.shape[-1])
            flat_x = x.reshape(-1, x.shape[-1])
            grad_B = flat_sgo.T @ (flat_x @ A.T)  # [out, r]

        return grad_input, grad_A, grad_B, None


class ALLoRALinear(nn.Module):
    """Drop-in replacement for a peft LoRA-wrapped nn.Linear's adapter
    branch: wraps a frozen base nn.Linear and adds an ALLoRA low-rank
    delta (base output + delta, matching how peft's Linear.forward()
    combines base_layer(x) + lora_B(lora_A(dropout(x))) * scaling --
    except here there is no dropout and no `scaling` multiplier, per
    ALLoRA's own removal of both).
    """

    def __init__(self, base_layer: nn.Linear, r: int, eta: float):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * (1.0 / r**0.5))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # B=0 init, same as standard LoRA
        self.eta = eta

    @property
    def bias(self):
        # Some HF model code inspects `.bias`/`.weight` directly on what
        # it expects to be a plain nn.Linear (e.g. wavlm's relative-
        # position-bias setup) even when not calling it as a Linear --
        # proxy through to the wrapped base layer rather than erroring,
        # since the ALLoRA delta branch has no bias of its own (matches
        # standard LoRA, which also only adds a bias-free low-rank delta).
        return self.base_layer.bias

    @property
    def weight(self):
        return self.base_layer.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        delta = _ALLoRAFunction.apply(x, self.lora_A, self.lora_B, self.eta)
        return base_out + delta
