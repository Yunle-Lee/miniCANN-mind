"""
NPU 优化模型层
在不修改原模型代码的前提下，通过 patch 方式注入 NPU 融合算子。
"""
import torch
import torch.nn.functional as F
from torch import nn
from .model_minimind import Attention, apply_rotary_pos_emb
from trainer.npu_optimizer import npu_fused_attention, npu_fused_rotary_pos_emb, _HAS_NPU


class NPUAttention(Attention):
    """
    替换 Attention.forward 使用 npu_fusion_attention。
    当 NPU 不可用或算子失败时自动回退到原始 SDPA。
    """

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # NPU 融合 RoPE
        cos, sin = position_embeddings
        if _HAS_NPU and self.flash:
            xq, xk = npu_fused_rotary_pos_emb(xq, xk, cos, sin)
        else:
            xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            self._repeat_kv(xk, self.n_rep).transpose(1, 2),
            self._repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # NPU 融合注意力 vs 标准 SDPA
        if _HAS_NPU and self.flash and (seq_len > 1) and past_key_value is None:
            output = npu_fused_attention(
                xq, xk, xv, self.n_local_heads,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal and attention_mask is None,
                attn_mask=attention_mask
            )
        elif self.flash and (seq_len > 1) and (attention_mask is None or torch.all(attention_mask == 1)) and (past_key_value is None or self.is_causal):
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal if past_key_value is None else (seq_len > 1)
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

    @staticmethod
    def _repeat_kv(x, n_rep):
        bs, slen, num_key_value_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
        )


def enable_npu_optimization(model):
    """
    将模型中的 Attention 层替换为 NPU 优化的版本。
    用法: model = enable_npu_optimization(model)
    """
    if not _HAS_NPU:
        return model

    for layer in model.model.layers:
        npu_attn = NPUAttention(model.config)
        npu_attn.load_state_dict(layer.self_attn.state_dict())
        npu_attn = npu_attn.to(model.device)
        layer.self_attn = npu_attn

    return model
