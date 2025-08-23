# image_search_core/utils.py
import torch
import hashlib

def get_device():
    """获取最优的可用计算设备 (CUDA > MPS > CPU)。"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def calculate_hash(filepath, chunk_size=8192):
    """计算文件的 SHA256 哈希值，用于检测文件内容变更。"""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (IOError, OSError):
        return None