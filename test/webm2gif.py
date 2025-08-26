import subprocess
import os
import shutil
import argparse
import sys

def check_ffmpeg_installed():
    """檢查系統是否安裝了 FFmpeg"""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("錯誤：找不到 FFmpeg 或 ffprobe。")
        print("請確保您已經安裝了 FFmpeg，並且它的路徑已經加入到系統環境變數中。")
        print("您可以從 https://ffmpeg.org/download.html 下載。")
        sys.exit(1)

def get_video_width(input_file):
    """使用 ffprobe 獲取影片的寬度"""
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        width_str = result.stdout.strip()
        if not width_str.isdigit():
            raise ValueError(f"無法解析寬度，ffprobe 輸出：'{width_str}'")
        return int(width_str)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"錯誤：獲取寬度失敗。\n{e}")
        return None


def convert_webm_to_gif(input_file, output_file, scale):
    """
    將 WebM 影片轉換為保留透明度的高品質 GIF。

    :param input_file: 輸入的 WebM 檔案路徑
    :param output_file: 輸出的 GIF 檔案路徑
    :param scale: 輸出 GIF 的縮放比例
    """
    check_ffmpeg_installed()

    if not os.path.exists(input_file):
        print(f"錯誤：輸入檔案 '{input_file}' 不存在。")
        return

    print(f"正在讀取影片 '{input_file}' 的資訊...")
    
    original_width = get_video_width(input_file)
    if not original_width:
        print("無法獲取影片寬度，轉換中止。")
        return
        
    # 計算目標寬度，並確保其為一個偶數，以獲得最佳相容性
    target_width = max(2, int(original_width * scale))
    if target_width % 2 != 0:
        target_width -= 1
        
    print(f"偵測到原始寬度為：{original_width}px")
    print(f"使用縮放比例：{scale}")
    print(f"輸出 GIF 目標寬度將為：{target_width}px\n")
    print("正在轉換為高品質 GIF...")

    # 定義濾鏡參數
    scale_filter = f"scale={target_width}:-1:flags=lanczos"
    paletteuse_options = "dither=bayer:bayer_scale=5:alpha_threshold=128"
    
    # 採用單一指令 filter_complex 流程，兼具速度與品質
    # 新增 stats_mode=single 來最佳化透明邊緣的調色盤，消除白邊
    filter_complex = (
        f"{scale_filter},split[s0][s1];"
        f"[s0]palettegen=stats_mode=single:reserve_transparent=on[p];"
        f"[s1][p]paletteuse={paletteuse_options}"
    )
    
    command_convert = [
        "ffmpeg",
        "-vcodec", "libvpx-vp9", # 明確指定 VP9 解碼器以支援透明度
        "-i", input_file,
        "-lavfi", filter_complex,
        "-y", # 覆蓋輸出檔案
        output_file,
    ]
    try:
        subprocess.run(command_convert, check=True, capture_output=True, text=True)
        print("-" * 30)
        print(f"🎉 轉換成功！高品質 GIF 已儲存至：{output_file}")
        print("-" * 30)
    except subprocess.CalledProcessError as e:
        print(f"錯誤：轉換失敗。FFmpeg 輸出:\n{e.stderr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="將 WebM 影片轉換為保留透明度的高品質 GIF。")
    parser.add_argument("input", help="輸入的 WebM 檔案路徑。")
    parser.add_argument("output", help="輸出的 GIF 檔案路徑。")
    parser.add_argument(
        "--scale", 
        type=float, 
        default=1.0, 
        help="輸出 GIF 的縮放比例，基於原始影片寬度計算。預設為 1.0。"
    )
    
    args = parser.parse_args()

    convert_webm_to_gif(args.input, args.output, args.scale)