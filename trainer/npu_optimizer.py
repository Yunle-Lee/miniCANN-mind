"""
Ascend NPU 算子优化层
提供 npu_fusion_attention 等华为融合算子的封装，带自动回退机制。
"""
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

_HAS_NPU = torch.npu.is_available()

def npu_fused_attention(query, key, value, num_heads, dropout_p=0.0, is_causal=False,
                        attn_mask=None, scale=None, input_layout="BNSD"):
    """
    使用 Ascend NPU 融合注意力算子（带 CUDA/CPU 回退）。

    参数:
        query: (B, N, S, D) BNSD 或 (B, S, N, D) BSND 格式
        input_layout: "BNSD" (默认) 或 "BSND"
        scale: softmax 缩放因子，默认 1/sqrt(D)
    返回:
        output: 与输入 query 相同格式
    """
    if not _HAS_NPU:
        return F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal)

    device = query.device
    if str(device) == 'cpu':
        return F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal)

    try:
        from torch_npu import npu_fusion_attention
        head_dim = query.size(-1)
        scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)
        attn_mask_tensor = None
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            attn_mask_tensor = attn_mask.to(torch.float16).mul_(-1e9)
        elif attn_mask is not None:
            attn_mask_tensor = attn_mask
        result = npu_fusion_attention(
            query, key, value, num_heads,
            input_layout=input_layout,
            pse=None, padding_mask=None, atten_mask=attn_mask_tensor,
            scale=scale,
            pre_tockens=65536,
            next_tockens=0 if is_causal else 65536,
            keep_prob=1.0 - dropout_p,
            inner_precise=0, sparse_mode=0
        )
        return result[0]
    except Exception as e:
        logger.debug(f"npu_fusion_attention 回退到 SDPA: {e}")
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=is_causal
        )


def npu_fused_rotary_pos_emb(q, k, cos, sin):
    """NPU 融合 RoPE (带回退)"""
    if not _HAS_NPU:
        return _apply_rotary_pos_emb_fallback(q, k, cos, sin)
    try:
        from torch_npu import npu_apply_rotary_pos_emb
        return npu_apply_rotary_pos_emb(q, k, cos, sin)
    except Exception:
        return _apply_rotary_pos_emb_fallback(q, k, cos, sin)


def _apply_rotary_pos_emb_fallback(q, k, cos, sin):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
    q_embed = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    k_embed = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
    return q_embed, k_embed


def npu_fast_gelu(x):
    """NPU 融合 GELU (带回退)"""
    if not _HAS_NPU:
        return F.gelu(x)
    try:
        from torch_npu import fast_gelu
        return fast_gelu(x)
    except Exception:
        return F.gelu(x)


class NPUConfig:
    """NPU 优化配置"""
    use_fusion_attention = True
    use_fused_rope = True
    use_fast_gelu = False
    use_compile = False
    graph_mode = False

    @classmethod
    def auto_config(cls):
        """根据硬件自动选择最优配置"""
        if not _HAS_NPU:
            cls.use_fusion_attention = False
            cls.use_fused_rope = False
            cls.use_fast_gelu = False
            return cls

        # 尝试验证 NPU 融合算子是否可用
        try:
            q = torch.randn(1, 1, 1, 1).npu()
            del q
            cls.use_fusion_attention = True
            cls.use_fused_rope = True
        except Exception:
            cls.use_fusion_attention = False
            cls.use_fused_rope = False

        return cls


def npu_memory_summary():
    """NPU 内存使用概况"""
    if not _HAS_NPU:
        return "NPU not available"
    lines = []
    for i in range(torch.npu.device_count()):
        try:
            t = torch.npu.get_device_properties(i)
            allocated = torch.npu.memory_allocated(i) / 1024**3
            cached = torch.npu.memory_reserved(i) / 1024**3
            lines.append(f"npu:{i} {t.name} | used: {allocated:.1f}GB | cached: {cached:.1f}GB | total: {t.total_mem/1024**3:.1f}GB")
        except Exception as e:
            lines.append(f"npu:{i} error: {e}")
    return "\n".join(lines)


def optimize_npu_performance():
    """NPU 性能优化建议"""
    if not _HAS_NPU:
        print("No NPU detected")
        return
    print("=" * 50)
    print("Ascend NPU 性能优化建议")
    print("=" * 50)
    print(f" NPU 数量: {torch.npu.device_count()}")
    for i in range(torch.npu.device_count()):
        try:
            p = torch.npu.get_device_properties(i)
            print(f" npu:{i} - {p.name} - {p.total_mem/1024**3:.1f}GB HBM")
        except Exception:
            pass
    print()
    print(" 推荐配置:")
    print("   - batch_size: 32 (单卡)")
    print("   - dtype: bfloat16")
    print("   - num_workers: 4 (避免 CPU 瓶颈)")
    print("   - accumulation_steps: 8")
    print("   - 启用 npu_fusion_attention")
    print("   - 启用 torch.compile (PyTorch 2.x)")
    print()
    print(" 环境变量:")
    print("   - ASCEND_VISIBLE_DEVICES=0,1   # 指定可见 NPU")
    print("   - HCCL_CONNECT_TIMEOUT=1800   # HCCL 超时")
    print("=" * 50)
