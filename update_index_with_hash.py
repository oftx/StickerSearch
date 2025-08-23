import os
import sys
import glob
import json
import hashlib
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

def get_device():
    """获取最优的可用计算设备。"""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def calculate_hash(filepath, chunk_size=8192):
    """计算文件的 SHA256 哈希值。"""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (IOError, OSError):
        return None # 文件无法访问

def load_model_and_processor(model_path, device):
    """从指定路径加载模型和处理器。"""
    try:
        model = ChineseCLIPModel.from_pretrained(model_path).to(device)
        processor = ChineseCLIPProcessor.from_pretrained(model_path, use_fast=True)
        return model, processor
    except OSError:
        print(f"错误：在路径 '{model_path}' 下找不到有效的模型文件。")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="通过哈希值检测变更，增量更新图片特征索引。")
    parser.add_argument('--image_dir', type=str, required=True, help="包含图片的文件夹路径。")
    parser.add_argument('--index_file', type=str, default="image_features.npz", help="要更新的索引文件路径。")
    args = parser.parse_args()

    # --- 1. 加载现有索引 ---
    indexed_features, indexed_paths, indexed_hashes = [], [], []
    if os.path.exists(args.index_file):
        print(f"步骤 1/5: 正在加载现有索引 '{args.index_file}'...")
        try:
            data = np.load(args.index_file)
            indexed_features = data['features']
            indexed_paths = data['paths']
            # 兼容没有哈希值的旧索引
            indexed_hashes = data['hashes'] if 'hashes' in data else [None] * len(indexed_paths)
        except Exception as e:
            print(f"警告：加载索引文件失败，将创建一个新索引。错误: {e}")
    else:
        print(f"步骤 1/5: 未找到索引文件，将创建一个新索引。")

    indexed_path_to_hash = {path: h for path, h in zip(indexed_paths, indexed_hashes)}

    # --- 2. 扫描当前文件夹并对比 ---
    print("步骤 2/5: 正在扫描文件夹并检测文件变更...")
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    current_paths = set()
    for ext in image_extensions:
        current_paths.update(glob.glob(os.path.join(args.image_dir, '**', ext), recursive=True))

    new_paths, modified_paths = [], []
    unchanged_paths = set()
    
    for path in tqdm(current_paths, desc="检测变更"):
        current_hash = calculate_hash(path)
        if not current_hash: continue

        if path not in indexed_path_to_hash:
            new_paths.append(path)
        elif current_hash != indexed_path_to_hash[path]:
            modified_paths.append(path)
        else:
            unchanged_paths.add(path)
    
    deleted_paths = set(indexed_path_to_hash.keys()) - current_paths

    if not new_paths and not modified_paths and not deleted_paths:
        print("\n文件无变化，索引已是最新，无需更新。")
        sys.exit(0)

    print(f"\n检测结果: {len(new_paths)} 新增, {len(modified_paths)} 修改, {len(deleted_paths)} 删除。")

    # --- 3. 准备更新数据 ---
    print("步骤 3/5: 正在准备更新数据...")
    # 保留未变更的文件
    final_features_list = []
    final_paths_list = []
    final_hashes_list = []
    
    if len(indexed_paths) > 0:
        for i, path in enumerate(indexed_paths):
            if path in unchanged_paths:
                final_features_list.append(indexed_features[i])
                final_paths_list.append(path)
                final_hashes_list.append(indexed_hashes[i])

    # --- 4. 为新增和修改的图片提取特征 ---
    paths_to_process = new_paths + modified_paths
    if paths_to_process:
        print("步骤 4/5: 正在为变更的图片提取特征...")
        device = get_device()
        print(f"  - 使用设备: {device}")
        
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            model_path = json.load(f).get("model_path")
            
        model, processor = load_model_and_processor(model_path, device)
        model.eval()

        with torch.no_grad():
            for path in tqdm(paths_to_process, desc="提取新特征"):
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    image_features = model.get_image_features(**inputs).cpu().numpy()
                    
                    final_features_list.append(image_features[0])
                    final_paths_list.append(path)
                    final_hashes_list.append(calculate_hash(path))
                except Exception as e:
                    print(f"\n警告：处理文件 {path} 时出错，已跳过。错误: {e}")

    # --- 5. 保存最终的索引 ---
    print("步骤 5/5: 正在保存更新后的索引...")
    if not final_paths_list:
        if os.path.exists(args.index_file):
            os.remove(args.index_file)
        print("\n图片库为空，已删除旧索引。")
    else:
        np.savez_compressed(
            args.index_file,
            features=np.array(final_features_list),
            paths=np.array(final_paths_list),
            hashes=np.array(final_hashes_list)
        )
        print(f"\n索引更新完成！现在共包含 {len(final_paths_list)} 张图片。")

if __name__ == "__main__":
    main()