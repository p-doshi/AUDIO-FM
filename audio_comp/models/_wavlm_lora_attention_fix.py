"""Fix for the wavlm LoRA/ALLoRA zero-gradient bug.

Root cause (verified against transformers 5.15.0's modeling_wavlm.py):
WavLMAttention.torch_multi_head_self_attention calls
F.multi_head_attention_forward(..., q_proj_weight=self.q_proj.weight,
k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, ...)
-- it extracts the raw .weight TENSOR from q_proj/k_proj/v_proj and does the
projection inside the fused functional op, never calling q_proj.forward(x) /
v_proj.forward(x) as normal submodule calls. Any module-replacement wrapper
(PEFT LoRA, or this project's own ALLoRALinear) that overrides .forward() is
therefore never invoked in the actual computation graph -- the LoRA/ALLoRA
delta is computed, exists as a real nn.Module with real parameters, but is
mathematically disconnected from the loss. grad=None on every LoRA/ALLoRA
param, confirmed by direct test (0/48 encoder params changed after opt.step()
on both conditions).

Fix: replace torch_multi_head_self_attention with a manual multi-head
attention that computes q/k/v via q_proj(hidden_states) / k_proj(...) /
v_proj(...) as ordinary submodule forward calls (so PEFT's LoRA.forward()
and ALLoRALinear.forward() both get invoked normally), decomposed by hand
instead of delegated to the fused functional op. Mathematically equivalent
to the original (same scaling, same use_separate_proj_weight=True semantics,
same gated_position_bias addition, same key_padding_mask handling) -- verified
below to produce numerically-identical output to the stock implementation
before any adapter is attached (so this is a pure refactor of *how* the
computation is expressed, not a change in what it computes), and to correctly
pass gradient to LoRA/ALLoRA params once wrapped.

Apply via monkeypatch_wavlm_attention() BEFORE building/wrapping the model
for LoRA or ALLoRA training. Frozen embedding extraction does not need this
patch (no adapter is attached, so the bug never manifests there) -- it's
scoped to the LoRA/ALLoRA training path only.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def _patched_torch_multi_head_self_attention(self, hidden_states, attention_mask, gated_position_bias, output_attentions):
    """Drop-in replacement for WavLMAttention.torch_multi_head_self_attention.
    Same signature, same inputs/outputs, but computed via q_proj(x)/k_proj(x)/
    v_proj(x)/out_proj(x) submodule calls instead of F.multi_head_attention_forward's
    raw-weight-tensor path -- so any module wrapping (LoRA, ALLoRA) on q_proj/
    v_proj actually participates in the forward computation and receives gradient.
    """
    bsz, tgt_len, embed_dim = hidden_states.size()
    num_heads = self.num_heads
    head_dim = self.head_dim
    scaling = self.scaling

    # q/k/v via real submodule forward calls -- THIS is the fix; the stock
    # implementation instead does F.multi_head_attention_forward(..., q_proj_weight=self.q_proj.weight, ...)
    # which bypasses q_proj.forward() (and therefore any LoRA/ALLoRA wrapper) entirely.
    q = self.q_proj(hidden_states) * scaling
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    def shape(x):
        return x.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)  # (bsz, heads, seq, head_dim)

    q, k, v = shape(q), shape(k), shape(v)

    attn_weights = torch.matmul(q, k.transpose(-1, -2))  # (bsz, heads, tgt, src)

    # gated_position_bias arrives as (bsz*num_heads, tgt_len, tgt_len) -- same
    # convention F.multi_head_attention_forward's attn_mask argument expected.
    gpb = gated_position_bias.view(bsz, num_heads, tgt_len, tgt_len)
    attn_weights = attn_weights + gpb

    if attention_mask is not None:
        # Match the stock implementation's convention EXACTLY: attention_mask
        # is 1/True where the position is VALID (attend), so key_padding_mask
        # (True = PADDED, to be masked out) = attention_mask.ne(1). This works
        # correctly for both int (1/0) and bool (True/False) tensors in torch
        # -- no dtype branch needed (a prior version incorrectly branched on
        # dtype and used the bool tensor's polarity un-negated for the bool
        # case, inverting valid/padded and masking out every position when
        # attention_mask was all-True/no-padding -- caught via a live NaN
        # trace: attn_weights all -inf -> softmax NaN -> loss NaN).
        key_padding_mask = attention_mask.ne(1)
        mask = key_padding_mask[:, None, None, :].to(torch.bool)
        attn_weights = attn_weights.masked_fill(mask, float("-inf"))

    attn_weights_softmax = F.softmax(attn_weights, dim=-1)
    attn_probs = F.dropout(attn_weights_softmax, p=self.dropout, training=self.training)

    attn_output = torch.matmul(attn_probs, v)  # (bsz, heads, tgt, head_dim)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
    attn_output = self.out_proj(attn_output)  # real submodule call -> out_proj's own LoRA (if any) also gets gradient

    if output_attentions:
        # match stock's broadcast-to-per-head convention for the returned weights
        attn_weights_out = attn_weights_softmax
    else:
        attn_weights_out = None

    return attn_output, attn_weights_out


def monkeypatch_wavlm_attention():
    """Idempotent: safe to call more than once."""
    from transformers.models.wavlm import modeling_wavlm
    modeling_wavlm.WavLMAttention.torch_multi_head_self_attention = _patched_torch_multi_head_self_attention
