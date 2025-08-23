import os
import sys
import glob
import json
import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

try:
    from transformers import ChineseCLIPProcessor, ChineseCLIPModel
except ImportError:
    print("错误：transformers 库未安装。")
    print("请运行 'pip install transformers torchvision' 进行安装。")
    sys.exit(1)

CONFIG_FILE = "config.json"

def load_model_and_processor(model_path, device):
    """从指定路径加载模型和处理器到指定设备。"""
    try:
        print(f"正在从 '{model_path}' 加载模型...")
        model = ChineseCLIPModel.from_pretrained(model_path).to(device)
        processor = ChineseCLIPProcessor.from_pretrained(model_path, use_fast=True)
        print("模型加载成功！")
        return model, processor
    except OSError:
        print(f"错误：在路径 '{model_path}' 下找不到有效的模型文件。")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="为图片文件夹创建特征索引。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help="包含图片的文件夹路径。"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default="image_features.npz",
        help="生成的索引文件路径 (默认为 image_features.npz)。"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help="手动指定模型路径，否则将从 model_config.json 读取。"
    )
    args = parser.parse_args()

    # 自动选择设备 (GPU优先)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"当前使用的设备: {device}")
    print(f"当前使用的设备: {device}")

    # --- 加载模型路径 ---
    model_path = args.model_path
    if not model_path:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                model_path = json.load(f).get("model_path")
        else:
            print(f"错误：找不到配置文件 '{CONFIG_FILE}'。请先运行 prepare_model.py。")
            sys.exit(1)

    # --- 加载模型 ---
    model, processor = load_model_and_processor(model_path, device)
    model.eval() # 设置为评估模式

    # --- 查找所有图片文件 ---
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    all_image_paths = []
    for ext in image_extensions:
        # glob支持递归搜索
        all_image_paths.extend(glob.glob(os.path.join(args.image_dir, '**', ext), recursive=True))
    
    if not all_image_paths:
        print(f"错误：在目录 '{args.image_dir}' 中未找到任何图片。")
        sys.exit(1)
        
    print(f"发现 {len(all_image_paths)} 张图片，开始提取特征...")

    # --- 提取特征 ---
    all_features = []
    all_paths = []

    with torch.no_grad():
        for path in tqdm(all_image_paths, desc="正在处理图片"):
            try:
                image = Image.open(path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                image_features = model.get_image_features(**inputs)
                
                # 将特征移动到CPU并转换为NumPy数组
                all_features.append(image_features.cpu().numpy())
                all_paths.append(path)
            except Exception as e:
                print(f"\n警告：处理文件 {path} 时出错，已跳过。错误: {e}")

    if not all_features:
        print("未能成功提取任何图片的特征。")
        sys.exit(1)

    # --- 保存到文件 ---
    # 将特征列表堆叠成一个大的NumPy数组
    features_array = np.vstack(all_features)
    
    # 使用 NumPy 的 savez_compressed 来高效保存
    np.savez_compressed(
        args.output_file,
        features=features_array,
        paths=np.array(all_paths) # 将路径列表也转换为NumPy数组
    )

    print(f"\n索引创建完成！ {len(all_paths)} 张图片的特征已保存到 '{args.output_file}'。")

if __name__ == "__main__":
    main()