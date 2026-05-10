"""
Ascend NPU 性能基准测试工具
测量模型在不同配置下的前向/反向速度、推理吞吐、融合算子加速比等。
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import argparse
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
from contextlib import nullcontext
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.npu_model import enable_npu_optimization
from trainer.huawei_adapter import get_device, device_count, get_amp_context, empty_cache
from trainer.npu_optimizer import npu_memory_summary, npu_fused_attention, _HAS_NPU

warnings.filterwarnings('ignore')

SUMMARY_WIDTH = 55


def print_header(title):
    print()
    print("=" * SUMMARY_WIDTH)
    print(f" {title}")
    print("=" * SUMMARY_WIDTH)


def print_result(name, value, unit=""):
    print(f"  {name:<40s} {value:>8.2f} {unit}")


@torch.no_grad()
def benchmark_forward(model, token_ids, attention_mask, n_warmup=10, n_iter=50, autocast_ctx=None):
    ctx = autocast_ctx or nullcontext()
    for _ in range(n_warmup):
        with ctx:
            _ = model(token_ids, attention_mask=attention_mask)
        empty_cache()
    torch.npu.synchronize() if _HAS_NPU else None

    start = time.time()
    for _ in range(n_iter):
        with ctx:
            _ = model(token_ids, attention_mask=attention_mask)
        empty_cache()
    torch.npu.synchronize() if _HAS_NPU else None
    elapsed = time.time() - start

    avg_ms = elapsed / n_iter * 1000
    tokens_per_sec = token_ids.numel() * n_iter / elapsed
    return avg_ms, tokens_per_sec


def benchmark_train_step(model, token_ids, labels, attention_mask, n_warmup=5, n_iter=20, autocast_ctx=None):
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    ctx = autocast_ctx or nullcontext()

    for _ in range(n_warmup):
        optimizer.zero_grad()
        with ctx:
            res = model(token_ids, attention_mask=attention_mask, labels=labels)
            loss = res.loss / 4
        loss.backward()
        optimizer.step()
        empty_cache()

    torch.npu.synchronize() if _HAS_NPU else None
    start = time.time()
    for _ in range(n_iter):
        optimizer.zero_grad()
        with ctx:
            res = model(token_ids, attention_mask=attention_mask, labels=labels)
            loss = res.loss / 4
        loss.backward()
        optimizer.step()
        empty_cache()
    torch.npu.synchronize() if _HAS_NPU else None
    elapsed = time.time() - start

    avg_ms = elapsed / n_iter * 1000
    tokens_per_sec = token_ids.numel() * n_iter / elapsed
    return avg_ms, tokens_per_sec


@torch.no_grad()
def benchmark_fusion_attention(batch_size=1, seq_len=512, num_heads=8, head_dim=96, n_iter=100):
    if not _HAS_NPU:
        return {}, {}
    device = f'{get_device()}:0'
    sdpa_times = []
    fusion_times = []

    q = torch.randn(batch_size, num_heads, seq_len, head_dim).half().to(device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim).half().to(device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim).half().to(device)

    for _ in range(10):
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
        npu_fused_attention(q, k, v, num_heads, is_causal=True)
    torch.npu.synchronize() if _HAS_NPU else None

    for _ in range(n_iter):
        t0 = time.time()
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.npu.synchronize()
        sdpa_times.append(time.time() - t0)

    for _ in range(n_iter):
        t0 = time.time()
        npu_fused_attention(q, k, v, num_heads, is_causal=True)
        torch.npu.synchronize()
        fusion_times.append(time.time() - t0)

    sdpa_avg = np.median(sdpa_times) * 1000
    fusion_avg = np.median(fusion_times) * 1000
    speedup = sdpa_avg / fusion_avg if fusion_avg > 0 else 0

    return {
        'SDPA (ms)': sdpa_avg,
        'Fusion (ms)': fusion_avg,
        'Speedup': speedup,
    }, {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'heads': num_heads,
        'head_dim': head_dim,
    }


@torch.no_grad()
def benchmark_inference(model, tokenizer, max_new_tokens=512, temperature=0.85, n_iter=10):
    prompts = ["请用Python写一个计算斐波那契数列的函数"]
    inputs = tokenizer(prompts, return_tensors="pt", truncation=True).to(next(model.parameters()).device)
    prompt_len = inputs.input_ids.shape[1]

    for _ in range(3):
        model.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=temperature)
    empty_cache()
    torch.npu.synchronize() if _HAS_NPU else None

    times = []
    tokens_list = []
    for _ in range(n_iter):
        t0 = time.time()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        torch.npu.synchronize() if _HAS_NPU else None
        elapsed = time.time() - t0
        times.append(elapsed)
        tokens_list.append(out.shape[1] - prompt_len)

    avg_time = np.mean(times)
    avg_tokens = np.mean(tokens_list)
    return {
        '生成耗时 (s)': avg_time,
        '生成 token 数': avg_tokens,
        '速度 (tokens/s)': avg_tokens / avg_time,
    }


def run_all_benchmarks(args):
    device = f'{get_device()}:0' if device_count() > 0 else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = get_amp_context(dtype=dtype) if device != 'cpu' else nullcontext()

    print_header("NPU 性能基准测试")
    print(f"  设备: {device}")
    print(f"  NPU 数量: {device_count()}")
    if _HAS_NPU:
        for i in range(torch.npu.device_count()):
            p = torch.npu.get_device_properties(i)
            total_mem = getattr(p, 'total_memory', getattr(p, 'total_mem', 0))
            print(f"  npu:{i} - {p.name} - {total_mem/1024**3:.1f}GB HBM")
    print(f"  数据类型: {args.dtype}")
    print(f"  模型: {args.hidden_size}dim, {args.num_hidden_layers}层, {'MoE' if args.use_moe else 'Dense'}")

    config = MiniMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe), flash_attn=True
    )

    print_header("1. 前向传播性能")
    for bs, seq_len in [(1, 128), (1, 512), (4, 128), (4, 512), (8, 128)]:
        model = MiniMindForCausalLM(config).half().to(device).eval()
        if args.npu_opt:
            model = enable_npu_optimization(model)
        x = torch.randint(0, 100, (bs, seq_len)).to(device)
        mask = torch.ones_like(x)
        avg_ms, tps = benchmark_forward(model, x, mask, autocast_ctx=autocast_ctx)
        print_result(f"BS={bs} seq={seq_len}", tps, "tokens/s")
        print_result(f"  单步耗时", avg_ms, "ms")
        del model, x, mask
        empty_cache()

    print_header("2. 训练 (前向+反向) 性能")
    for bs, seq_len in [(1, 128), (2, 128), (4, 128)]:
        model = MiniMindForCausalLM(config).half().to(device).train()
        if args.npu_opt:
            model = enable_npu_optimization(model)
        x = torch.randint(0, 100, (bs, seq_len)).to(device)
        y = x.clone()
        mask = torch.ones_like(x)
        avg_ms, tps = benchmark_train_step(model, x, y, mask, autocast_ctx=autocast_ctx)
        print_result(f"BS={bs} seq={seq_len}", tps, "tokens/s")
        print_result(f"  单步耗时", avg_ms, "ms")
        del model, x, y, mask
        empty_cache()

    if _HAS_NPU and args.bench_fusion:
        print_header("3. 融合注意力加速比 (NPU)")
        for seq_len in [128, 512, 1024]:
            results, params = benchmark_fusion_attention(
                batch_size=1, seq_len=seq_len,
                num_heads=8, head_dim=96, n_iter=50
            )
            print(f"  Seq_len={seq_len}:")
            print(f"    SDPA:     {results['SDPA (ms)']:.3f} ms")
            print(f"    Fusion:   {results['Fusion (ms)']:.3f} ms")
            print(f"    Speedup:  {results['Speedup']:.2f}x")

    if _HAS_NPU:
        print_header("4. NPU 内存使用")
        print(npu_memory_summary())

    if args.bench_inference:
        print_header("5. 推理性能")
        model = MiniMindForCausalLM(config).half().to(device).eval()
        if args.npu_opt:
            model = enable_npu_optimization(model)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        results = benchmark_inference(model, tokenizer, max_new_tokens=args.max_gen_len, n_iter=5)
        for k, v in results.items():
            print_result(k, v)
        del model, tokenizer
        empty_cache()

    if args.npu_opt and _HAS_NPU:
        print_header("6. NPU 优化 vs 原版对比")
        model_base = MiniMindForCausalLM(config).half().to(device).eval()
        model_opt = MiniMindForCausalLM(config).half().to(device).eval()
        model_opt = enable_npu_optimization(model_opt)
        x = torch.randint(0, 100, (2, 256)).to(device)
        mask = torch.ones_like(x)
        base_ms, base_tps = benchmark_forward(model_base, x, mask, autocast_ctx=autocast_ctx)
        opt_ms, opt_tps = benchmark_forward(model_opt, x, mask, autocast_ctx=autocast_ctx)
        print_result(f"原版 (SDPA)", base_tps, "tokens/s")
        print_result(f"优化 (Fusion)", opt_tps, "tokens/s")
        print_result(f"加速比", opt_tps / base_tps, "x")
        del model_base, model_opt, x, mask
        empty_cache()

    print()
    print("=" * SUMMARY_WIDTH)
    print(" 基准测试完成")
    print("=" * SUMMARY_WIDTH)


def main():
    parser = argparse.ArgumentParser(description="Ascend MiniMind NPU 基准测试")
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--dtype', default='bfloat16', choices=['float16', 'bfloat16'])
    parser.add_argument('--npu_opt', default=1, type=int, choices=[0, 1], help='启用 NPU 融合算子')
    parser.add_argument('--bench_fusion', default=1, type=int, choices=[0, 1], help='测试融合注意力加速比')
    parser.add_argument('--bench_inference', default=0, type=int, choices=[0, 1], help='测试推理速度')
    parser.add_argument('--max_gen_len', default=256, type=int)
    parser.add_argument('--tokenizer_path', default='model', type=str)
    args = parser.parse_args()

    run_all_benchmarks(args)


if __name__ == '__main__':
    main()
