# Ascend MiniMind 从零训练教程

本教程逐步演示如何在**华为昇腾 Ascend910 NPU** 上，从零训练一个 64M 参数的小型语言模型。从环境配置到推理对话，每个阶段都有完整的命令和说明。

---

## 目录

1. [环境搭建](#1-环境搭建)
2. [数据准备](#2-数据准备)
3. [第 1 阶段：预训练（Pretrain）](#3-第-1-阶段预训练pretrain)
4. [第 2 阶段：指令微调（SFT）](#4-第-2-阶段指令微调sft)
5. [第 3 阶段：LoRA 微调（可选）](#5-第-3-阶段lora-微调可选)
6. [推理与对话](#6-推理与对话)
7. [多卡分布式训练](#7-多卡分布式训练)
8. [性能调优建议](#8-性能调优建议)
9. [常见问题](#9-常见问题)

---

## 1. 环境搭建

### 1.1 硬件要求

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| NPU | Ascend 910 (1卡) | Ascend 910 (2卡) |
| NPU 显存 | 32GB HBM | 64GB HBM |
| 内存 | 32GB | 64GB+ |
| 磁盘 | 50GB | 100GB |

### 1.2 检查 CANN 与驱动

```bash
# 查看 CANN 安装目录
ls /home/developer/Ascend/

# 查看 NPU 状态
npu-smi info

# 查看驱动版本
npu-smi info -t board -i 0
```

正常输出应显示 `Ascend910` 芯片，`Health: OK`。

### 1.3 验证 PyTorch + torch_npu

```bash
python3 -c "
import torch, torch_npu
print('PyTorch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('NPU 可用:', torch.npu.is_available())
print('NPU 数量:', torch.npu.device_count())
for i in range(torch.npu.device_count()):
    print(f'  npu:{i} - {torch.npu.get_device_name(i)}')
"
```

预期输出：
```
PyTorch: 2.7.1
torch_npu: 2.7.1.post2
NPU 可用: True
NPU 数量: 2
  npu:0 - Ascend910
  npu:1 - Ascend910
```

### 1.4 安装项目依赖

```bash
cd /mnt/workspace/huawei-minimind

# 安装核心依赖
pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# 如遇 transformers 版本冲突，可锁定版本
pip3 install transformers==4.57.6 datasets==3.6.0
```

### 1.5 验证项目可运行

```bash
cd /mnt/workspace/huawei-minimind
python3 -c "
import sys
sys.path.insert(0, '.')

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.huawei_adapter import get_device, device_count
from dataset.lm_dataset import PretrainDataset
from transformers import AutoTokenizer

# 1) 确认设备
dev = get_device()
print(f'检测到设备: {dev}, 数量: {device_count()}')

# 2) 创建模型
config = MiniMindConfig()
model = MiniMindForCausalLM(config)
print(f'模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

# 3) 移动模型到 NPU
model = model.half().npu()
print(f'模型已加载到: {next(model.parameters()).device}')

# 4) 前向传播测试
import torch
x = torch.randint(0, 100, (1, 32)).npu()
with torch.no_grad():
    out = model(x)
    print(f'前向传播通过, logits shape: {out.logits.shape}')

# 5) 分词器测试
tokenizer = AutoTokenizer.from_pretrained('model')
tokens = tokenizer('你好, 世界!')
print(f'分词器测试通过, 词表大小: {tokenizer.vocab_size}')
print(f'   \"你好, 世界!\" -> ids: {tokens.input_ids}')
"
```

---

## 2. 数据准备

### 2.1 下载数据

数据文件来自 ModelScope，需要下载到项目的 `dataset/` 目录。

#### 方式 A：用 modelscope 库下载（推荐）

```bash
pip3 install modelscope

python3 -c "
from modelscope import snapshot_download
snapshot_download('gongjy/minimind_dataset', local_dir='./dataset')
"
```

#### 方式 B：手动下载

从 [ModelScope - gongjy/minimind_dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 下载以下文件到 `./dataset/`：

| 文件 | 用途 | 大小 |
|------|------|------|
| `pretrain_t2t_mini.jsonl` | 预训练数据 | ~50MB |
| `sft_t2t_mini.jsonl` | SFT 微调数据 | ~5MB |
| `lora_medical.jsonl` | LoRA 医疗数据 | ~1MB |

### 2.2 验证数据格式

```bash
cd /mnt/workspace/huawei-minimind

# 检查预训练数据
head -3 dataset/pretrain_t2t_mini.jsonl | python3 -c "
import sys, json
for i, line in enumerate(sys.stdin):
    d = json.loads(line)
    print(f'第{i+1}条: text={d[\"text\"][:80]}...')
"

# 检查 SFT 数据
python3 -c "
import json
with open('dataset/sft_t2t_mini.jsonl') as f:
    d = json.loads(f.readline())
    for conv in d['conversations']:
        print(f'{conv[\"role\"]}: {conv[\"content\"][:50]}...')
"
```

### 2.3 测试数据加载

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset, SFTDataset

tokenizer = AutoTokenizer.from_pretrained('model')

# 测试预训练数据集
pretrain_ds = PretrainDataset('dataset/pretrain_t2t_mini.jsonl', tokenizer, max_length=340)
x, y = pretrain_ds[0]
print(f'预训练数据: input_ids shape={x.shape}, labels shape={y.shape}')

# 测试 SFT 数据集
sft_ds = SFTDataset('dataset/sft_t2t_mini.jsonl', tokenizer, max_length=768)
x, y = sft_ds[0]
print(f'SFT 数据: input_ids shape={x.shape}, labels shape={y.shape}')
"
```

---

## 3. 第 1 阶段：预训练（Pretrain）

预训练是从头训练语言模型的第一步。在此阶段，模型学习语言的统计规律。

### 3.1 快速开始

```bash
cd /mnt/workspace/huawei-minimind/trainer

python train_pretrain.py \
    --batch_size 32 \
    --accumulation_steps 8 \
    --epochs 2 \
    --max_seq_len 340 \
    --hidden_size 768 \
    --num_hidden_layers 8
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 32 | 单卡 batch size |
| `--accumulation_steps` | 8 | 梯度累积步数（等效 batch = 32×8 = 256） |
| `--epochs` | 2 | 训练轮数 |
| `--max_seq_len` | 340 | 序列截断长度 |
| `--hidden_size` | 768 | 模型隐藏层维度 |
| `--num_hidden_layers` | 8 | Transformer 层数 |
| `--learning_rate` | 5e-4 | 初始学习率 |
| `--dtype` | bfloat16 | 混合精度类型 |
| `--use_moe` | 0 | 是否使用 MoE 架构 |
| `--use_compile` | 0 | 是否使用 torch.compile |

### 3.2 显存估算

| 配置 | 显存占用 | 建议 |
|------|---------|------|
| batch=16, seq=340 | ~8GB | 单卡可跑 |
| batch=32, seq=340 | ~12GB | 推荐配置 |
| batch=64, seq=340 | ~20GB | 大 batch |

### 3.3 训练监控

训练过程中会实时输出 loss 变化：

```
Epoch:[1/2](100/8000), loss: 5.2341, logits_loss: 5.2341, aux_loss: 0.0000, lr: 0.00045000, epoch_time: 45.2min
Epoch:[1/2](200/8000), loss: 4.8912, logits_loss: 4.8912, aux_loss: 0.0000, lr: 0.00040000, epoch_time: 40.1min
```

- **loss**: 总 loss（logits_loss + aux_loss）
- **logits_loss**: 主 loss（语言建模 head）
- **aux_loss**: MoE 辅助 loss（仅 MoE 模型有）
- **lr**: 当前学习率（余弦退火）

### 3.4 中断后恢复训练

```bash
python train_pretrain.py \
    --from_resume 1 \
    --batch_size 32 \
    --epochs 2 \
    --max_seq_len 340
```

程序会自动检测 `checkpoints/` 目录下的断点文件，从上次位置继续训练。

### 3.5 训练产出

训练完成后，在 `out/` 目录下生成权重文件：

```
out/pretrain_768.pth          # 模型权重
checkpoints/pretrain_768_resume.pth  # 断点（含优化器状态）
```

---

## 4. 第 2 阶段：指令微调（SFT）

SFT 让模型学会遵循人类指令进行对话。

### 4.1 基于预训练权重微调

```bash
cd /mnt/workspace/huawei-minimind/trainer

python train_full_sft.py \
    --from_weight pretrain \
    --batch_size 16 \
    --epochs 2 \
    --max_seq_len 768 \
    --learning_rate 1e-5 \
    --hidden_size 768 \
    --num_hidden_layers 8
```

**关键参数差异（与预训练对比）：**

| 参数 | 预训练 | SFT |
|------|--------|-----|
| `--learning_rate` | 5e-4 | **1e-5** |
| `--max_seq_len` | 340 | **768** |
| `--accumulation_steps` | 8 | **1** |
| `--from_weight` | none | **pretrain** |

### 4.2 仅训练若干步验证

```bash
cd /mnt/workspace/huawei-minimind/trainer

python train_full_sft.py \
    --from_weight pretrain \
    --batch_size 2 \
    --max_seq_len 128 \
    --epochs 1 \
    --save_interval 5 \
    --log_interval 1
```

此命令使用极小配置验证代码是否正确运行，预计 1-2 分钟完成。

### 4.3 训练产出

```
out/full_sft_768.pth           # SFT 权重
checkpoints/full_sft_768_resume.pth  # 断点
```

---

## 5. 第 3 阶段：LoRA 微调（可选）

LoRA 只训练少量额外参数，适合快速适配特定领域。

```bash
cd /mnt/workspace/huawei-minimind/trainer

python train_lora.py \
    --lora_name lora_medical \
    --data_path ../dataset/lora_medical.jsonl \
    --from_weight full_sft \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-4
```

LoRA 参数量仅占总参数的 ~0.5%：

```
LLM Total Params: 63.91M
LoRA Params: 0.33M
LoRA Ratio: 0.52%
```

---

## 6. 推理与对话

### 6.1 快速测试

```bash
cd /mnt/workspace/huawei-minimind

python eval_llm.py --weight full_sft
```

选择 `[0] 自动测试` 可运行预设问题列表；选择 `[1] 手动输入` 进入交互模式。

### 6.2 指定不同权重

```bash
# 使用预训练权重
python eval_llm.py --weight pretrain --max_new_tokens 256

# 使用 SFT 权重
python eval_llm.py --weight full_sft

# 使用 SFT + LoRA 权重
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

### 6.3 启用长文本推理

如果需要在推理时处理超过训练长度的文本，启用 YaRN RoPE 外推：

```bash
python eval_llm.py --weight full_sft --inference_rope_scaling --max_new_tokens 16384
```

### 6.4 修改配置测试小模型

```bash
# 使用更小的模型配置（3072 词表 + 4 层 + 512 隐藏维度）
python train_pretrain.py --hidden_size 512 --num_hidden_layers 4 --max_seq_len 256 --batch_size 64

# 推理这个小模型
python eval_llm.py --hidden_size 512 --num_hidden_layers 4 --weight pretrain
```

---

## 7. 多卡分布式训练

### 7.1 单机多卡（DDP）

```bash
cd /mnt/workspace/huawei-minimind/trainer

# 2 卡分布式预训练
python -m torch.distributed.run \
    --nproc_per_node 2 \
    train_pretrain.py \
    --batch_size 32 \
    --accumulation_steps 4 \
    --epochs 2

# 2 卡分布式 SFT
python -m torch.distributed.run \
    --nproc_per_node 2 \
    train_full_sft.py \
    --from_weight pretrain \
    --batch_size 16 \
    --epochs 2
```

### 7.2 多卡加速比

| NPU 数量 | Batch Size | 单步时间 | 加速比 |
|----------|-----------|---------|--------|
| 1 | 32 | 1.0x | 1.0x |
| 2 | 64 | ~0.55x | ~1.8x |
| 4 | 128 | ~0.3x | ~3.3x |

### 7.3 环境变量

```bash
# 指定可见的 NPU 设备
export ASCEND_VISIBLE_DEVICES=0,1
python -m torch.distributed.run --nproc_per_node 2 train_pretrain.py

# 只使用单卡
export ASCEND_VISIBLE_DEVICES=2,3
python -m torch.distributed.run --nproc_per_node 2 train_pretrain.py

# 查看环境变量
echo $ASCEND_VISIBLE_DEVICES
```

---

## 8. 性能调优建议

### 8.1 混合精度

Ascend 910 对 bfloat16 有原生支持，比 float16 更快且数值更稳定：

```bash
# bfloat16（推荐）
python train_pretrain.py --dtype bfloat16

# float16（备选）
python train_pretrain.py --dtype float16
```

### 8.2 DataLoader 优化

```bash
# NPU 上建议减小 num_workers（避免 CPU 瓶颈）
python train_pretrain.py --num_workers 4

# 增大 batch_size 提高吞吐
python train_pretrain.py --batch_size 64 --accumulation_steps 4
```

### 8.3 梯度累积

对于大模型显存不足的情况，减少 batch_size 并增加 accumulation_steps：

```bash
# 等效 batch = 16 × 16 = 256
python train_pretrain.py --batch_size 16 --accumulation_steps 16
```

### 8.4 CANN 配置

```bash
# 可选：设置 CANN 性能配置
export HCCL_CONNECT_TIMEOUT=1800
```

---

## 9. 常见问题

### Q1: `torch.npu.is_available()` 返回 False

**可能原因：**
- CANN 驱动未正确加载 → 运行 `npu-smi info` 检查 NPU 状态
- `torch_npu` 未安装或版本不匹配 → `pip3 list | grep torch_npu`
- 环境变量未设置 → `export ASCEND_VISIBLE_DEVICES=0,1`

### Q2: 显存不足（OOM）

**解决方案：**
```bash
# 减小 batch_size
python train_pretrain.py --batch_size 8 --accumulation_steps 32

# 缩短序列长度
python train_pretrain.py --max_seq_len 256

# 使用更小的模型（减少隐藏层维度）
python train_pretrain.py --hidden_size 512 --num_hidden_layers 6
```

### Q3: 训练 loss 不下降

**检查项：**
- 学习率是否合适（预训练 5e-4, SFT 1e-5）
- 数据路径是否正确 → 检查 `--data_path` 指向的文件是否存在
- 权重是否正确加载 → `--from_weight pretrain` 的文件是否存在

### Q4: 多卡训练报错

**可能原因：**
- HCCL 通信初始化失败 → 检查 `ASCEND_VISIBLE_DEVICES` 设置
- 进程间通信超时 → `export HCCL_CONNECT_TIMEOUT=1800`

### Q5: 如何清理缓存

```bash
# 清理 torch_npu 缓存
python3 -c "import torch; torch.npu.empty_cache()"

# 清理磁盘上的检查点
rm -rf /mnt/workspace/huawei-minimind/checkpoints/*.pth
```

### Q6: 自定义数据格式

**预训练数据格式（JSONL）：**
```json
{"text": "这是训练语料的第一条数据。词林郁郁，学海无涯。"}
{"text": "这是第二条数据。机器学习是人工智能的核心。"}
```

**SFT 数据格式（JSONL）：**
```json
{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}]}
{"conversations": [{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "1+1=2"}]}
```

---

## 完整训练流水线（一键脚本）

```bash
cd /mnt/workspace/huawei-minimind

# 第1步：预训练（约2-4小时）
cd trainer && python train_pretrain.py \
    --batch_size 32 \
    --accumulation_steps 8 \
    --epochs 2 \
    --save_dir ../out

# 第2步：SFT 微调（约1-2小时）
python train_full_sft.py \
    --from_weight pretrain \
    --batch_size 16 \
    --epochs 2 \
    --save_dir ../out

# 第3步：LoRA 微调（可选，约10分钟）
python train_lora.py \
    --lora_name lora_medical \
    --from_weight full_sft \
    --epochs 10

# 第4步：推理测试
cd .. && python eval_llm.py --weight full_sft
```

---

> 文档版本：v1.0 | 适用环境：Ascend 910 / CANN 8.5 / PyTorch 2.7 / torch_npu 2.7
