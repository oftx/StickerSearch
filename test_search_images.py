import os
import sys
import json
import requests
import argparse
from PIL import Image
import torch

# 动态导入，以防 transformers 未安装
try:
    from transformers import ChineseCLIPProcessor, ChineseCLIPModel
except ImportError:
    print("错误：transformers 库未安装。")
    print("请运行 'pip install transformers torchvision' 进行安装。")
    sys.exit(1)

CONFIG_FILE = "config.json"

def load_model_and_processor(model_path):
    """从指定路径加载模型和处理器。"""
    try:
        print(f"正在从 '{model_path}' 加载模型...")
        model = ChineseCLIPModel.from_pretrained(model_path)
        processor = ChineseCLIPProcessor.from_pretrained(model_path, use_fast=True)
        print("模型加载成功！")
        return model, processor
    except OSError:
        print(f"错误：在路径 '{model_path}' 下找不到有效的模型文件。")
        print(f"请确认路径是否正确，或重新运行 prepare_model.py 来下载模型。")
        sys.exit(1)

def load_images(image_paths):
    """从一系列本地路径或 URL 加载图片。"""
    images = []
    valid_paths = []
    print("\n正在加载图片...")
    for path in image_paths:
        try:
            if path.startswith(('http://', 'https://')):
                response = requests.get(path, stream=True)
                response.raise_for_status()
                img = Image.open(response.raw).convert("RGB")
            else:
                img = Image.open(path).convert("RGB")
            images.append(img)
            valid_paths.append(path)
            print(f"  - 成功加载: {path}")
        except Exception as e:
            print(f"  - 警告：无法加载图片 '{path}'，已跳过。原因: {e}")
    return images, valid_paths

def search_best_image(model, processor, images, query, image_paths):
    """使用模型进行语义搜索，找出最匹配的图片。"""
    if not images:
        print("\n没有有效的图片可供搜索。")
        return

    print(f"\n正在使用 “{query}” 进行语义搜索...")
    try:
        # 关键：一次性处理所有图片和单个文本
        inputs = processor(text=[query], images=images, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
        
        # logits_per_image 的形状是 [图片数量, 文本数量]
        # 在这里是 [图片数量, 1]，所以我们需要压缩一下
        logits_per_image = outputs.logits_per_image.squeeze()
        
        # 使用 softmax 将分数转换为概率分布
        probs = logits_per_image.softmax(dim=0)
        
        # 找到分数最高的那个的索引
        best_image_index = torch.argmax(probs).item()
        best_image_path = image_paths[best_image_index]
        
        print("\n--- 搜索结果 ---")
        print(f"🏆 最佳匹配图片: {best_image_path}")
        
        print("\n--- 各图片匹配度详情 ---")
        for i, path in enumerate(image_paths):
            indicator = "🏆" if i == best_image_index else "  "
            print(f"{indicator} {path}: {probs[i].item():.2%}")

    except Exception as e:
        print(f"在模型推理过程中发生错误: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="使用中文语义在多张图片中进行搜索。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        nargs='+', # 允许接收一个或多个图片参数
        help="要搜索的图片路径或 URL 列表，用空格分隔。"
    )
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='用于搜索的中文描述。例如: "一个女孩在海边微笑"'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help="手动指定模型所在的目录路径。\n如果未指定，脚本将自动从 model_config.json 读取。"
    )
    args = parser.parse_args()

    model_path = args.model_path

    if not model_path:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                model_path = config.get("model_path")
        else:
            print(f"错误：找不到模型配置文件 '{CONFIG_FILE}'。")
            print("请先运行 'python prepare_model.py' 来下载模型并生成配置。")
            sys.exit(1)

    if not os.path.isdir(model_path):
        print(f"错误：配置的模型路径 '{model_path}' 不存在或不是一个目录。")
        print("请检查 model_config.json 或手动指定的路径是否正确。")
        sys.exit(1)

    # 执行核心流程
    model, processor = load_model_and_processor(model_path)
    images, valid_paths = load_images(args.images)
    search_best_image(model, processor, images, args.query, valid_paths)

if __name__ == "__main__":
    main()