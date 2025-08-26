# image_search_core/converter.py

import subprocess
import shutil
from PIL import Image

def check_ffmpeg_installed():
    """检查系统是否安装了 FFmpeg"""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("错误：找不到 FFmpeg 或 ffprobe。请确保已安装 FFmpeg 并将其添加到系统路径中。")

def _get_video_width(input_file):
    """使用 ffprobe 获取影片的宽度"""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width", "-of", "default=noprint_wrappers=1:nokey=1", input_file
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        width_str = result.stdout.strip()
        if not width_str.isdigit():
            raise ValueError(f"无法解析宽度，ffprobe 输出：'{width_str}'")
        return int(width_str)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"错误：获取宽度失败。\n{e}")
        return None

def convert_webm_to_gif(source_path, output_path, params):
    """
    将 WebM 影片转换为保留透明度的高品质 GIF。
    :param source_path: 输入的 WebM 文件路径
    :param output_path: 输出的 GIF 文件路径
    :param params: 包含转换参数的字典，如 'scale', 'algorithm' 等
    """
    check_ffmpeg_installed()

    original_width = _get_video_width(source_path)
    if not original_width:
        raise RuntimeError("无法获取视频宽度，转换中止。")
    
    # 从 params 获取参数，并提供默认值
    scale = float(params.get('scale', 1.0))
    algorithm = params.get('algorithm', 'lanczos')
    reserve_transparent = bool(params.get('reserve_transparent', True))
    alpha_threshold = int(params.get('alpha_threshold', 128))

    target_width = max(2, int(original_width * scale))
    # 确保宽度为偶数，某些编码器需要
    if target_width % 2 != 0:
        target_width -= 1

    scale_filter = f"scale={target_width}:-1:flags={algorithm}"
    
    # --- 核心修改：根据 reserve_transparent 参数动态构建 filter_complex ---
    palettegen_options = "stats_mode=single"
    if reserve_transparent:
        palettegen_options += ":reserve_transparent=on"
        paletteuse_options = f"dither=bayer:bayer_scale=5:alpha_threshold={alpha_threshold}"
    else:
        paletteuse_options = "dither=bayer:bayer_scale=5" # 不使用 alpha_threshold

    filter_complex = (
        f"{scale_filter},split[s0][s1];"
        f"[s0]palettegen={palettegen_options}[p];"
        f"[s1][p]paletteuse={paletteuse_options}"
    )
    
    command_convert = [
        "ffmpeg", "-vcodec", "libvpx-vp9", "-i", source_path,
        "-lavfi", filter_complex, "-y", output_path
    ]
    try:
        # 使用 utf-8 编码来处理可能的路径或错误信息
        result = subprocess.run(command_convert, check=True, capture_output=True, text=True, encoding='utf-8')
        return {"status": "success", "message": "转换成功", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"转换失败。FFmpeg 输出:\n{e.stderr}")

def convert_webp(source_path, output_path, target_format, params):
    """
    转换 WebP 图片到 PNG, JPG 或 GIF。
    :param source_path: 输入的 WebP 文件路径
    :param output_path: 输出的文件路径
    :param target_format: 'png', 'jpeg', 或 'gif'
    :param params: 包含转换参数的字典，如 'scale'
    """
    try:
        with Image.open(source_path) as img:
            # 确保图像是 RGBA 模式以保留透明度
            if img.mode != 'RGBA' and target_format.lower() != 'jpeg':
                img = img.convert('RGBA')

            # 处理缩放
            scale = float(params.get('scale', 1.0))
            if scale != 1.0:
                original_width, original_height = img.size
                target_width = int(original_width * scale)
                target_height = int(original_height * scale)
                img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

            save_params = {}
            if target_format.lower() == 'jpeg':
                # 对于 JPEG，需要转换为 RGB
                if img.mode == 'RGBA':
                    # 创建一个白色背景，然后将带 alpha 通道的图像粘贴上去
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.getchannel('A'))
                    img = background
                else:
                    img = img.convert('RGB')
                save_params['quality'] = 95 # 设置 JPEG 保存质量
            
            img.save(output_path, format=target_format.upper(), **save_params)
        
        return {"status": "success", "message": "转换成功"}
    except Exception as e:
        raise RuntimeError(f"处理 WebP 文件时出错: {e}")