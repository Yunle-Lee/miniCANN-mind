# Ascend MiniMind

> 华为昇腾 NPU 上的轻量 LLM 训练项目，基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 适配

将 64M 参数的小型语言模型训练流水线完整迁移至 **Ascend 910 NPU**，支持 Pretrain → SFT → LoRA 全流程。

## 环境

- 2× Ascend910 NPU（64GB HBM / 卡）
- CANN 8.5 + PyTorch 2.7 + torch_npu 2.7
- Python 3.11

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
cd trainer

# 预训练
python train_pretrain.py --batch_size 32 --epochs 2

# 指令微调
python train_full_sft.py --from_weight pretrain --batch_size 16 --epochs 2

# 推理
cd .. && python eval_llm.py --weight full_sft
```

## 文档

完整教程请查阅 **[TUTORIAL.md](./TUTORIAL.md)**，包含：

- [x] 环境搭建与验证
- [x] 数据准备（ModelScope 下载）
- [x] 预训练从零开始
- [x] SFT 指令微调
- [x] LoRA 高效微调
- [x] 分布式多卡训练
- [x] 推理与对话测试
- [x] 性能调优建议
- [x] 常见问题排查

## 与原始版差异

| 项目 | MiniMind | Ascend MiniMind |
|------|----------|-----------------|
| 设备 | CUDA only | **NPU + CUDA** 自动适配 |
| AMP | `torch.cuda.amp` | `torch.npu.amp` / `torch.cuda.amp` |
| 分布式 | NCCL | **HCCL**（NPU）/ NCCL |
| 适配层 | 无 | `trainer/huawei_adapter.py`（设备无关，~60行） |

## 模型架构

- 64M 参数 LLaMA-like Decoder-Only
- GQA (8 Q-heads, 4 KV-heads), RoPE, SwiGLU
- 可选 MoE（198M-A64M，单 token 激活 64M）
- 6400 BPE 词表（含中英文）
