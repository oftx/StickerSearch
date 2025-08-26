import os
import ctypes
import win32clipboard
from typing import Union, List
from contextlib import contextmanager

@contextmanager
def open_clipboard():
    """一个上下文管理器，用于安全地打开和关闭Windows剪贴板。"""
    win32clipboard.OpenClipboard()
    try:
        yield
    finally:
        win32clipboard.CloseClipboard()

def copy_files_to_clipboard(paths: Union[str, List[str]]):
    """
    将一个或多个文件复制到Windows剪贴板。
    这允许将文件粘贴到文件资源管理器或其他支持文件粘贴的应用程序中。

    :param paths: 单个文件路径 (str) 或文件路径列表 (List[str])。
                  路径可以使用正斜杠 (/) 或反斜杠 (\\)。
    """
    # 此环境变量可在某些情况下帮助避免UAC（用户账户控制）弹窗，但可能并非必需，
    # 且可能有副作用。请谨慎使用。
    os.environ.update({"__COMPAT_LAYER": "RUnAsInvoker"})

    # 确保`paths`是一个列表
    if isinstance(paths, str):
        paths = [paths]
    
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        raise TypeError("输入必须是字符串或字符串列表。")

    # 将所有路径转换为使用Windows反斜杠的绝对路径
    # CF_HDROP格式要求使用绝对路径
    abs_paths = [os.path.abspath(p).replace('/', '\\') for p in paths]

    # 定义DROPFILES结构体
    class DROPFILES(ctypes.Structure):
        _fields_ = [
            ("pFiles", ctypes.c_uint),
            ("x", ctypes.c_long),
            ("y", ctypes.c_long),
            ("fNC", ctypes.c_int),
            ("fWide", ctypes.c_bool),
        ]

    # 准备剪贴板数据
    # 数据格式是一个由空字符分隔的文件路径列表，并以两个空字符结尾
    # 编码为UTF-16LE
    pDropFiles = DROPFILES()
    pDropFiles.pFiles = ctypes.sizeof(DROPFILES)
    pDropFiles.fWide = True  # 使用Unicode (UTF-16)

    # 用'\0'连接路径，并在末尾添加两个'\0'
    files_str = "\0".join(abs_paths)
    # [2:]用于移除BOM（字节顺序标记）
    data = files_str.encode("U16")[2:] + b"\0\0"
    
    # 将结构体和路径数据合并
    clipboard_data = bytes(pDropFiles) + data

    try:
        # 使用with语句安全地操作剪贴板
        with open_clipboard():
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_HDROP, clipboard_data)
    except Exception as e:
        # 提供更详细的错误信息
        print(f"复制文件到剪贴板时出错: {e}")

if __name__ == '__main__':
    copy_files_to_clipboard(r"d:\Desktop\image.gif")