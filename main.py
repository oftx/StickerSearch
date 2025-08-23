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
    print("请运行 'pip install transformers' 进行安装。")
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

def load_image(image_path):
    """从本地路径或 URL 加载图片。"""
    try:
        if image_path.startswith(('http://', 'https://')):
            # 从 URL 加载
            response = requests.get(image_path, stream=True)
            response.raise_for_status() # 如果请求失败则抛出异常
            image = Image.open(response.raw)
        else:
            # 从本地文件加载
            image = Image.open(image_path)
        return image.convert("RGB")
    except Exception as e:
        print(f"错误：无法加载图片 '{image_path}'。")
        print(f"错误详情: {e}")
        sys.exit(1)

def classify_image(model, processor, image, texts):
    """使用模型对图片和文本进行分类。"""
    print("\n正在进行图文匹配...")
    try:
        # 使用处理器对图像和文本进行预处理
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        
        # 将处理后的数据输入模型
        with torch.no_grad(): # 推理时不需要计算梯度
            outputs = model(**inputs)
        
        # logits_per_image 表示图片与每个文本描述的相似度分数
        logits_per_image = outputs.logits_per_image
        
        # 使用 softmax 将分数转换为概率
        probs = logits_per_image.softmax(dim=1).squeeze() # squeeze() 移除不必要的维度

        print("\n--- 分析结果 ---")
        for text, prob in zip(texts, probs):
            print(f"图片与 “{text}” 的匹配概率: {prob.item():.2%}")
        
    except Exception as e:
        print(f"在模型推理过程中发生错误: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="使用 Chinese-CLIP 模型进行零样本图像分类。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help="要分析的图片路径或 URL。"
    )
    parser.add_argument(
        '--texts',
        type=str,
        required=True,
        help='候选的文本描述，用英文逗号 "," 分隔。\n例如: "一只猫的照片,一只狗的照片,风景画"'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help="手动指定模型所在的目录路径。\n如果未指定，脚本将自动从 model_config.json 读取。"
    )
    args = parser.parse_args()

    model_path = args.model_path

    # 如果未手动指定路径，则从配置文件读取
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

    # 解析文本
    texts = [text.strip() for text in args.texts.split(',')]
    if len(texts) < 2:
        print("错误：请至少提供两个用逗号分隔的文本描述。")
        sys.exit(1)

    # 执行核心流程
    model, processor = load_model_and_processor(model_path)
    image = load_image(args.image)
    classify_image(model, processor, image, texts)

if __name__ == "__main__":
    main()