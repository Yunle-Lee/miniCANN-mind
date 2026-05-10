"""
ModelScope 数据下载脚本
从 ModelScope 下载 Minimind 训练数据到本地 dataset/ 目录。
支持断点续传和完整性校验。
"""
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
MODELSCOPE_REPO = 'gongjy/minimind_dataset'
MODELSCOPE_URL = f'https://www.modelscope.cn/datasets/{MODELSCOPE_REPO}.git'

REQUIRED_FILES = {
    'pretrain_t2t_mini.jsonl': '预训练数据',
    'sft_t2t_mini.jsonl': 'SFT 指令微调数据',
    'lora_medical.jsonl': 'LoRA 医疗数据',
}

ADVANCED_FILES = {
    'dpo.jsonl': 'DPO 偏好对齐数据',
    'rlaif.jsonl': 'RLAIF 强化学习数据',
    'agent.jsonl': 'Agent 工具调用数据',
}


def check_existing_files(file_dict):
    """检查已有文件，返回缺失列表"""
    os.makedirs(DATASET_DIR, exist_ok=True)
    existing = []
    missing = []
    partial = []
    for fname, desc in file_dict.items():
        fpath = os.path.join(DATASET_DIR, fname)
        if os.path.exists(fpath):
            fsize = os.path.getsize(fpath)
            if fsize > 0:
                existing.append((fname, desc, fsize))
            else:
                partial.append((fname, desc))
        else:
            missing.append((fname, desc))
    return existing, missing, partial


def download_modelscope(force=False):
    """使用 modelscope 库下载"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("[ERROR] modelscope 库未安装。运行: pip install modelscope")
        return False

    print(f"正在从 ModelScope 下载: {MODELSCOPE_REPO}")
    print(f"  目标目录: {DATASET_DIR}")
    os.makedirs(DATASET_DIR, exist_ok=True)

    try:
        snapshot_download(MODELSCOPE_REPO, local_dir=DATASET_DIR, resume_download=True)
        print(f"  下载完成!")
        return True
    except Exception as e:
        print(f"  ModelScope 下载失败: {e}")
        return False


def download_git(force=False):
    """使用 git clone 作为备用"""
    import subprocess
    print(f"正在通过 git clone 下载: {MODELSCOPE_URL}")
    print(f"  目标目录: {DATASET_DIR}")

    try:
        tmp_dir = DATASET_DIR + '_tmp'
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', MODELSCOPE_URL, tmp_dir],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"  git clone 失败: {result.stderr[:200]}")
            return False

        import shutil
        for f in os.listdir(tmp_dir):
            if f.endswith('.jsonl'):
                shutil.move(os.path.join(tmp_dir, f), os.path.join(DATASET_DIR, f))
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  下载完成!")
        return True
    except Exception as e:
        print(f"  git clone 失败: {e}")
        return False


def verify_files():
    """验证文件完整性"""
    print("\n文件完整性检查:")
    all_ok = True
    all_files = {**REQUIRED_FILES, **ADVANCED_FILES}
    for fname, desc in all_files.items():
        fpath = os.path.join(DATASET_DIR, fname)
        if os.path.exists(fpath):
            fsize = os.path.getsize(fpath)
            try:
                with open(fpath, 'r') as f:
                    line_count = sum(1 for _ in f)
                print(f"  [OK] {fname} ({desc}) - {fsize/1024/1024:.1f}MB, {line_count} 条记录")
            except Exception as e:
                print(f"  [ERR] {fname} 读取失败: {e}")
                all_ok = False
        else:
            all_ok = False
    return all_ok


def download_files(files_to_download, force=False):
    """下载指定的文件列表"""
    if not files_to_download:
        print("所有文件已存在，无需下载。")
        return verify_files()

    print(f"需要下载 {len(files_to_download)} 个文件:")
    for fname, desc in files_to_download:
        print(f"  - {fname} ({desc})")

    if download_modelscope(force=force):
        pass
    elif download_git(force=force):
        pass
    else:
        print("\n所有下载方式均失败。")
        print("请手动从以下地址下载文件并放入 dataset/ 目录:")
        print(f"  https://www.modelscope.cn/datasets/{MODELSCOPE_REPO}/files")
        return False

    return verify_files()


def main():
    parser = argparse.ArgumentParser(description="下载 MiniMind 训练数据")
    parser.add_argument('--force', action='store_true', help='强制重新下载')
    parser.add_argument('--include-advanced', action='store_true', help='包括 DPO/RLAIF/Agent 高级数据')
    parser.add_argument('--verify-only', action='store_true', help='仅检查文件完整性')
    args = parser.parse_args()

    print("=" * 50)
    print("MiniMind 数据下载工具")
    print("=" * 50)

    target_files = dict(REQUIRED_FILES)
    if args.include_advanced:
        target_files.update(ADVANCED_FILES)

    existing, missing, partial = check_existing_files(target_files)

    if existing:
        print("\n已存在的文件:")
        for fname, desc, fsize in existing:
            print(f"  [OK] {fname} ({desc}) - {fsize/1024/1024:.1f}MB")

    if partial:
        print("\n不完整的文件:")
        for fname, desc in partial:
            print(f"  [PARTIAL] {fname} ({desc}) - 重新下载")

    if args.verify_only:
        return verify_files()

    all_missing = missing + partial
    if args.force:
        all_missing = [(f, d) for f, d in target_files.items()]

    if args.force or all_missing:
        download_files(all_missing, force=args.force)
    else:
        print("\n所有文件就绪，无需下载。")
        verify_files()

    print("\n" + "=" * 50)
    print("提示: 运行训练脚本前请确认 dataset/ 目录下有对应数据文件。")
    print("=" * 50)


if __name__ == '__main__':
    main()
