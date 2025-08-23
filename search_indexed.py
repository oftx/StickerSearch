import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

try:
    from transformers import ChineseCLIPProcessor, ChineseCLIPModel
except ImportError:
    print("错误：transformers 库未安装。")
    print("请运行 'pip install transformers torchvision' 进行安装。")
    sys.exit(1)

CONFIG_FILE = "config.json"

def load_model_and_processor(model_path, device):
    """从指定路径加载模型和处理器。"""
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
        description="在预先计算好的图片索引中进行语义搜索。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='用于搜索的中文描述。'
    )
    parser.add_argument(
        '--index_file',
        type=str,
        default="image_features.npz",
        help="要加载的索引文件路径 (默认为 image_features.npz)。"
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help="显示最匹配的前 K 个结果 (默认为 5)。"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help="手动指定模型路径，否则将从 model_config.json 读取。"
    )
    args = parser.parse_args()

    if not os.path.exists(args.index_file):
        print(f"错误：索引文件 '{args.index_file}' 不存在。")
        print("请先运行 python index_images.py --image_dir <你的图片目录> 来创建索引。")
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"当前使用的设备: {device}")

    # --- 加载模型路径 ---
    model_path = args.model_path
    if not model_path:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            model_path = json.load(f).get("model_path")

    # --- 加载模型 (仅用于处理文本) ---
    model, processor = load_model_and_processor(model_path, device)
    model.eval()

    # --- 加载索引数据 ---
    print(f"正在从 '{args.index_file}' 加载索引...")
    data = np.load(args.index_file)
    image_features = torch.from_numpy(data['features']).to(device)
    image_paths = data['paths']
    print(f"索引加载完毕，包含 {len(image_paths)} 张图片。")

    # --- 处理文本查询 ---
    with torch.no_grad():
        inputs = processor(text=[args.query], return_tensors="pt").to(device)
        text_features = model.get_text_features(**inputs)

    # --- 计算相似度 (核心步骤) ---
    # 1. 归一化 (L2 Normalization)，这是计算余弦相似度的关键
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    # 2. 计算点积 (Dot Product)，等价于归一化后的余弦相似度
    # [1, 512] @ [512, N] -> [1, N]
    similarity_scores = (text_features @ image_features.T).squeeze()

    # --- 查找 Top K 结果 ---
    top_k = min(args.top_k, len(image_paths))
    top_results = torch.topk(similarity_scores, k=top_k)
    
    scores = top_results.values.cpu().numpy()
    indices = top_results.indices.cpu().numpy()

    print(f"\n--- “{args.query}” 的 Top {top_k} 搜索结果 ---")
    for i in range(top_k):
        score = scores[i]
        path = image_paths[indices[i]]
        print(f"  {i+1}. 匹配度: {score:.4f} - 图片: {path}")

if __name__ == "__main__":
    main()