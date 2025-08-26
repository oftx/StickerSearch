# image_search_core/utils.py
import torch
import hashlib
import platform
import os
import subprocess

# --- 新增 Windows 平台所需的导入 ---
# 只有在 Windows 平台上才尝试导入，避免在 macOS/Linux 上因缺少库而报错
system_platform = platform.system()
if system_platform == "Windows":
    import ctypes
    import win32clipboard
    from contextlib import contextmanager

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

# =================================================================
# --- 新增：平台原生的文件拷贝功能 ---
# =================================================================

def _copy_file_to_clipboard_mac(file_path: str) -> (bool, str):
    """在 macOS 上使用 AppleScript 将文件对象复制到剪贴板。"""
    try:
        script = f'set the clipboard to (POSIX file "{file_path}")'
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
        return True, f"文件 '{os.path.basename(file_path)}' 已拷贝。"
    except subprocess.CalledProcessError as e:
        error_message = f"AppleScript 执行失败: {e.stderr.decode()}"
        return False, error_message
    except Exception as e:
        return False, f"复制文件时发生未知错误: {e}"

def _copy_file_to_clipboard_windows(file_path: str) -> (bool, str):
    """在 Windows 上使用 CF_HDROP 格式将文件对象复制到剪贴板。"""
    
    @contextmanager
    def open_clipboard_context():
        win32clipboard.OpenClipboard()
        try:
            yield
        finally:
            win32clipboard.CloseClipboard()

    class DROPFILES(ctypes.Structure):
        _fields_ = [
            ("pFiles", ctypes.c_uint),
            ("x", ctypes.c_long),
            ("y", ctypes.c_long),
            ("fNC", ctypes.c_int),
            ("fWide", ctypes.c_bool),
        ]

    abs_path = os.path.abspath(file_path).replace('/', '\\')
    
    pDropFiles = DROPFILES()
    pDropFiles.pFiles = ctypes.sizeof(DROPFILES)
    pDropFiles.fWide = True

    data = abs_path.encode("U16")[2:] + b"\0\0"
    clipboard_data = bytes(pDropFiles) + data

    try:
        with open_clipboard_context():
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_HDROP, clipboard_data)
        return True, f"文件 '{os.path.basename(file_path)}' 已拷贝。"
    except Exception as e:
        return False, f"复制文件到剪贴板时出错: {e}"


def copy_file_to_clipboard(file_path: str) -> (bool, str):
    """
    根据当前操作系统，将指定的文件对象复制到系统剪贴板。

    返回一个元组 (success: bool, message: str)。
    """
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        return False, f"错误：文件 '{abs_path}' 未找到。"

    system = platform.system()
    if system == "Darwin":  # Darwin 是 macOS 的内核名
        return _copy_file_to_clipboard_mac(abs_path)
    elif system == "Windows":
        return _copy_file_to_clipboard_windows(abs_path)
    else:
        return False, f"不支持的操作系统: {system}。此功能仅在 Windows 和 macOS 上可用。"