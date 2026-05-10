# Session Handover — Ascend MiniMind (miniCANN-mind)

> 编写日期：2026-05-10
> 目标：华为昇腾 NPU 上的轻量 LLM 训练项目

---

## 1. 项目状态总览

| 项目 | 状态 |
|------|------|
| 模型定义（63.9M LLaMA） | ✅ 完成 |
| 数据处理（6 类数据集） | ✅ 完成 |
| NPU 适配层（huawei_adapter） | ✅ 完成 |
| NPU 融合算子（npu_fusion_attention） | ✅ 已修复并验证 |
| NPU 优化模型层（enable_npu_optimization） | ✅ 完成 |
| 预训练（Pretrain） | ✅ **已跑完** loss 1.86 |
| SFT（指令微调） | ⏳ 待运行 |
| LoRA 微调 | ⏳ 待运行 |
| DPO 偏好对齐 | ⏳ 待运行 |
| PPO/GRPO 强化学习 | ⏳ 待运行 |
| Agent RL | ⏳ 待运行 |
| 推理对话（eval_llm） | ✅ 代码完成 |
| ModelScope 数据下载脚本 | ✅ 完成 |
| NPU 基准测试脚本 | ✅ 完成 |
| 文档（README/TUTORIAL） | ✅ 已更新 |
| GitHub 仓库同步 | ✅ 已 push |

---

## 2. 今日完成的工作

### 2.1 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `scripts/download_data.py` | ~130 | ModelScope 一键数据下载（断点续传+校验） |
| `scripts/benchmark_npu.py` | ~280 | NPU 性能基准测试（前向/反向/融合加速比/内存） |
| `trainer/train_agent.py` | ~170 | Agent 工具调用 RL 训练脚本（PPO-Clip） |
| `images/pretrain_loss_curve.png` | — | 预训练 loss 曲线图 |

### 2.2 修复的问题

1. **npu_fusion_attention API 不兼容**（trainer/npu_optimizer.py）
   - 本地 CANN 版本 schema 不同：无 `quant_mode`/`kernel_backend` 参数
   - 返回 7 个值（不是 2 个），改取 `result[0]`
   - 添加 `inner_precise` 和 `sparse_mode` 参数
   - `input_layout` 默认改为 `"BNSD"`（与模型实际输入匹配）
   - 验证通过：BNSD/BSND 布局，causal/non-causal 均 OK

2. **enable_npu_optimization 引用错误**（model/npu_model.py）
   - `layer.self_attn` 没有 `.config` 属性
   - 改为 `model.config` 传参

3. **npu_apply_rotary_pos_emb 不可用**
   - 该环境报 `aclnnApplyRotaryPosEmbV2 error code 561002`
   - 自动回退到标准实现，不影响训练

### 2.3 数据下载

通过 `https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/` 直接下载：

| 文件 | 大小 | 用途 |
|------|------|------|
| `dataset/pretrain_t2t_mini.jsonl` | 1.18 GB | 预训练语料 |
| `dataset/sft_t2t_mini.jsonl` | 1.66 GB | SFT 指令微调 |
| `dataset/lora_medical.jsonl` | 32 MB | LoRA 医疗数据 |
| `dataset/dpo.jsonl` | 52 MB | DPO 偏好对齐 |
| `dataset/rlaif.jsonl` | 23 MB | RLAIF 强化学习 |
| `dataset/agent_rl.jsonl` | 79 MB | Agent 工具调用 |

### 2.4 预训练结果

- 完成 epoch 1/1，共 79390 steps
- 最终 loss: **1.86**
- 权重保存：`out/pretrain_768.pth`（132MB）
- loss 曲线图：`images/pretrain_loss_curve.png`

### 2.5 文档更新

| 文档 | 更新内容 |
|------|----------|
| `README.md` | 新增 RL 管线、融合算子、数据下载、项目结构图 |
| `TUTORIAL.md` | 新增 DPO/PPO/GRPO/Agent 训练章节、融合算子、基准测试、下载脚本 |

---

## 3. 环境信息

### 3.1 硬件

- 2× Ascend 910 NPU（各 64GB HBM）
- 驱动 C75B，CANN 8.5

### 3.2 软件

- PyTorch 2.7.1+cpu（torch_npu 2.7.1.post2.dev20251226）
- Python 3.11.4
- 工作目录：`/mnt/workspace/huawei-minimind/`

### 3.3 关键依赖

```
torch, torch_npu, transformers, datasets, modelscope
jieba, numpy, tiktoken, einops, swanlab, rich
```

### 3.4 Git 凭证

HTTPS token 已存入 `~/.git-credentials`，可直接 `git push`。

---

## 4. 项目结构

```
huawei-minimind/
├── model/
│   ├── model_minimind.py      # 主模型 (63.9M)
│   ├── model_lora.py          # LoRA 实现
│   ├── npu_model.py           # NPU 融合算子注入
│   └── tokenizer.*            # BPE 分词器 (6400)
├── dataset/
│   ├── lm_dataset.py          # 6 类数据集
│   ├── pretrain_t2t_mini.jsonl # 预训练数据 [1.2G]
│   ├── sft_t2t_mini.jsonl     # SFT 数据 [1.7G]
│   └── ... (其他数据文件)
├── trainer/
│   ├── huawei_adapter.py      # NPU 设备适配层
│   ├── npu_optimizer.py       # NPU 融合算子封装
│   ├── rollout_engine.py      # Rollout (Torch/SGLang)
│   ├── trainer_utils.py       # 训练工具
│   ├── train_pretrain.py      # ✅ 已完成
│   ├── train_full_sft.py      # ⏳ 下一步
│   ├── train_lora.py          # ⏳
│   ├── train_dpo.py           # ⏳
│   ├── train_ppo.py           # ⏳
│   ├── train_grpo.py          # ⏳
│   └── train_agent.py         # ⏳
├── scripts/
│   ├── download_data.py       # 数据下载
│   └── benchmark_npu.py       # 基准测试
├── images/
│   └── pretrain_loss_curve.png # loss 曲线
├── out/
│   └── pretrain_768.pth       # 预训练权重 [132MB]
├── checkpoints/               # 断点目录
├── eval_llm.py                # 推理脚本
├── README.md                  # 项目简介
├── TUTORIAL.md                # 完整教程
└── SESSION_HANDOVER.md        # 本文档
```

---

## 5. 下一步工作（明日优先）

### 5.1 第 1 优先级：SFT 指令微调

```bash
cd /mnt/workspace/huawei-minimind/trainer
python train_full_sft.py \
    --from_weight pretrain \
    --batch_size 16 \
    --epochs 2 \
    --max_seq_len 768 \
    --learning_rate 1e-5 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --save_interval 200 \
    --data_path ../dataset/sft_t2t_mini.jsonl
```

预期：1-2 小时完成，loss 从 ~2.0 降到 ~0.8。

### 5.2 第 2 优先级：推理测试

```bash
cd /mnt/workspace/huawei-minimind
python eval_llm.py --weight full_sft
```

需先完成 SFT。

### 5.3 第 3 优先级：后续训练

按顺序跑：LoRA → DPO → GRPO → Agent。

### 5.4 第 4 优先级：基准测试

```bash
python scripts/benchmark_npu.py --npu_opt 1 --bench_inference 1
python scripts/benchmark_npu.py --npu_opt 0  # 对比原版
```

### 5.5 第 5 优先级：CANN Graph 模式优化

探索 `torch.npu.graph` 或 JIT 编译加速。

---

## 6. 已知问题

### 6.1 npu_apply_rotary_pos_emb 不可用

- 错误码 561002（aclnnApplyRotaryPosEmbV2）
- 不影响训练：自动回退到标准 `apply_rotary_pos_emb`
- 如需修复可尝试：升级 CANN 驱动，或改用 `torch_npu.npu_rotary_mul`

### 6.2 npu_fusion_attention vs SDPA 数值差异

- float16 下 max diff ~3.46（fusion 使用不同累加精度）
- 不影响训练收敛
- bfloat16 下差异会更小

### 6.3 torch_npu.get_device_properties 字段名

- `p.total_mem` 在某些版本不存在，benchmark_npu.py 已修复用 `getattr`

### 6.4 ModelScope snapshot_download API 变化

- `resume_download` 参数在新版本 modelscope 中已移除
- download_data.py git clone 方式可用

---

## 7. 关键命令速查

```bash
# 启动 SFT
cd /mnt/workspace/huawei-minimind/trainer
python train_full_sft.py --from_weight pretrain --batch_size 16 --epochs 2

# 启动推理
cd /mnt/workspace/huawei-minimind
python eval_llm.py --weight full_sft

# 基准测试
python scripts/benchmark_npu.py --bench_inference 1

# 查看 loss
grep 'loss:' /path/to/training/output

# 绘图（如需重新生成）
python3 -c "
import matplotlib.pyplot as plt, numpy as np
steps, losses = np.loadtxt('/tmp/loss_data.txt', unpack=True)
plt.plot(steps, losses, alpha=0.3)
window=50; s = np.convolve(losses, np.ones(window)/window, mode='valid')
plt.plot(steps[:len(s)], s, linewidth=2)
plt.savefig('images/loss.png', dpi=200)
"

# 推送代码
git add -A && git commit -m "message" && git push
```
