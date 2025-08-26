import subprocess
import os

def copy_file_object_mac(file_path):
    """在 macOS 上将文件对象复制到剪贴板。"""
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f"错误：文件 '{abs_path}' 未找到。")
        return
        
    try:
        # 使用 AppleScript 告诉 Finder 将文件放入剪贴板
        script = f'set the clipboard to (POSIX file "{abs_path}")'
        subprocess.run(["osascript", "-e", script], check=True)
        print(f"文件对象 '{abs_path}' 已成功复制到剪贴板。")
    except Exception as e:
        print(f"复制文件对象时出错: {e}")

# --- 使用示例 ---
copy_file_object_mac("/Users/user/Dev/copytest/image.gif")