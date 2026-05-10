"""
Huawei Ascend NPU compatibility adapter for MiniMind.
Provides device-agnostic wrappers so training scripts work on both NPU and CUDA.
"""
import torch
import os

def get_device():
    if torch.npu.is_available():
        return "npu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def device_count():
    if torch.npu.is_available():
        return torch.npu.device_count()
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def set_device(rank):
    if torch.npu.is_available():
        torch.npu.set_device(rank)
    else:
        torch.cuda.set_device(rank)

def manual_seed(seed):
    if torch.npu.is_available():
        torch.npu.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def manual_seed_all(seed):
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)

def empty_cache():
    if torch.npu.is_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()

def get_dist_backend():
    return "hccl" if torch.npu.is_available() else "nccl"

def get_amp_context(dtype=None, device_type=None):
    if device_type is None:
        device_type = get_device()
    if dtype is None:
        dtype = torch.bfloat16
    if device_type == "npu":
        return torch.npu.amp.autocast(dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)

def get_grad_scaler(enabled=True):
    if torch.npu.is_available():
        return torch.npu.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)
