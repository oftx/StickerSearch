# image_search_core/indexer.py
import os
import glob
import json
import random
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import cv2

from .model_loader import ModelLoader
from .utils import calculate_hash
from .config import IMAGE_EXTENSIONS, DEFAULT_INDEX_FILE, CONFIG_FILE

class ImageIndexer:
    """
    封装了创建和更新图片特征索引的所有逻辑。
    """
    def __init__(self, image_dir: str, index_file: str = DEFAULT_INDEX_FILE, model_path: str = None):
        if not os.path.isdir(image_dir):
            raise ValueError(f"提供的图片目录不存在: {image_dir}")
        self.image_dir = image_dir
        self.index_file = index_file
        self.model_loader = ModelLoader(model_path)
        self.device = self.model_loader.device

    def _save_image_dir_to_config(self):
        """将当前索引的图片目录绝对路径保存到配置文件中。"""
        config_data = {}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                try:
                    config_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"警告: {CONFIG_FILE} 文件格式错误，将重新创建。")

        config_data['image_base_dir'] = os.path.abspath(self.image_dir)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        print(f"图片目录 '{config_data['image_base_dir']}' 已保存到 '{CONFIG_FILE}'")

    def update(self):
        """主方法，执行完整的索引更新流程。"""
        indexed_data = self._load_or_initialize_index()
        new_paths, modified_paths, deleted_paths = self._scan_and_compare(indexed_data)

        summary = {
            "new": len(new_paths),
            "modified": len(modified_paths),
            "deleted": len(deleted_paths),
            "total": 0,
            "message": ""
        }

        if not new_paths and not modified_paths and not deleted_paths:
            summary["total"] = len(indexed_data.get('paths', []))
            summary["message"] = "文件无变化，索引已是最新。"
            print(summary["message"])
            return summary

        print(f"\n检测结果: {len(new_paths)} 新增, {len(modified_paths)} 修改, {len(deleted_paths)} 删除。")
        final_data = self._process_changes(indexed_data, new_paths, modified_paths, deleted_paths)
        self._save_index(final_data)
        self._save_image_dir_to_config()

        summary["total"] = len(final_data.get('paths', []))
        summary["message"] = f"索引更新成功！新增 {summary['new']}, 修改 {summary['modified']}, 删除 {summary['deleted']}。当前索引共 {summary['total']} 张图片。"
        return summary

    def _load_or_initialize_index(self):
        """
        加载现有索引。
        为保证内部处理一致性，无论文件中存储的是相对还是绝对路径，
        本方法加载后一律转换成绝对路径供程序内部使用。
        """
        if not os.path.exists(self.index_file):
            print(f"未找到索引文件，将创建一个新索引。")
            return {"features": [], "paths": [], "hashes": [], "base_dir": None}

        print(f"正在加载现有索引 '{self.index_file}'...")
        try:
            data = np.load(self.index_file, allow_pickle=True)
            base_dir = str(data['base_dir'][0]) if 'base_dir' in data else None
            resolved_paths = [os.path.abspath(p) for p in data['paths']]

            return {
                "features": list(data['features']),
                "paths": resolved_paths,
                "hashes": list(data['hashes']) if 'hashes' in data else [None] * len(resolved_paths),
                "base_dir": base_dir
            }
        except Exception as e:
            print(f"警告：加载索引文件失败，将创建一个新索引。错误: {e}")
            return {"features": [], "paths": [], "hashes": [], "base_dir": None}

    def _scan_and_compare(self, indexed_data: dict):
        """
        扫描磁盘文件，并与索引数据比对。
        本方法内部统一使用绝对路径进行比较，以确保准确性。
        """
        indexed_path_to_hash = {path: h for path, h in zip(indexed_data['paths'], indexed_data['hashes'])}

        current_paths = set()
        for ext in IMAGE_EXTENSIONS:
            found_files = glob.glob(os.path.join(self.image_dir, '**', ext), recursive=True)
            for path in found_files:
                current_paths.add(os.path.abspath(path))

        new, modified = [], []
        for path in tqdm(current_paths, desc="检测文件变更"):
            current_hash = calculate_hash(path)
            if not current_hash: continue
            if path not in indexed_path_to_hash:
                new.append(path)
            elif current_hash != indexed_path_to_hash[path]:
                modified.append(path)

        deleted = set(indexed_path_to_hash.keys()) - current_paths
        return new, modified, list(deleted)

    def _process_changes(self, indexed_data, new_paths, modified_paths, deleted_paths):
        # 1. 移除已删除和已修改的条目
        final_data = {"features": [], "paths": [], "hashes": []}
        paths_to_remove = set(modified_paths + deleted_paths)
        for i, path in enumerate(indexed_data['paths']):
            if path not in paths_to_remove:
                final_data["features"].append(indexed_data["features"][i])
                final_data["paths"].append(path)
                final_data["hashes"].append(indexed_data["hashes"][i])

        # 2. 为新增和修改的图片提取特征并添加
        paths_to_process = new_paths + modified_paths
        if not paths_to_process:
            return final_data

        print("正在为变更的图片提取特征...")
        model, processor = self.model_loader.load()
        with torch.no_grad():
            for path in tqdm(paths_to_process, desc="提取新特征"):
                try:
                    image = None
                    path_lower = path.lower()

                    # 如果是视频或GIF，使用OpenCV提取一个随机帧
                    if path_lower.endswith(('.webm', '.mp4', '.gif')):
                        cap = cv2.VideoCapture(path)
                        if cap.isOpened():
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            # 只有在视频/GIF有多于1帧时才进行随机选择
                            if total_frames > 0:
                                # 生成一个 0 到 total_frames-1 之间的随机索引
                                random_frame_index = random.randint(0, total_frames - 1)
                                # 设置视频捕获对象到随机帧的位置
                                cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
                                
                                ret, frame = cap.read()
                                if ret:
                                    # OpenCV 读取的颜色通道是 BGR, Pillow/模型需要 RGB
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    image = Image.fromarray(frame_rgb)
                            cap.release()
                        else:
                             print(f"\n警告：无法使用 OpenCV 打开文件 {path}，已跳过。")
                             continue
                    # 否则，作为普通静态图片处理
                    else:
                        image = Image.open(path).convert("RGB")

                    if image is None:
                        print(f"\n警告：未能从 {path} 加载图像或帧，已跳过。")
                        continue

                    inputs = processor(images=image, return_tensors="pt").to(self.device)
                    features = model.get_image_features(**inputs).cpu().numpy()[0]
                    final_data["features"].append(features)
                    final_data["paths"].append(path)
                    final_data["hashes"].append(calculate_hash(path))
                except Exception as e:
                    print(f"\n警告：处理文件 {path} 时出错，已跳过。错误: {e}")
        return final_data

    def _save_index(self, final_data: dict):
        """
        保存最终的索引数据到文件。
        根据 image_dir 是否在当前工作目录下，决定使用相对路径或绝对路径保存。
        """
        if not final_data["paths"]:
            if os.path.exists(self.index_file):
                os.remove(self.index_file)
            print("\n图片库为空，已删除旧索引。")
            return

        print(f"正在保存更新后的索引，共 {len(final_data['paths'])} 张图片...")

        abs_image_dir = os.path.abspath(self.image_dir)
        current_dir = os.getcwd()

        is_in_current_folder = abs_image_dir.startswith(current_dir + os.sep) or abs_image_dir == current_dir

        if is_in_current_folder:
            print("图片目录在当前工作目录下，使用相对路径保存。")
            stored_base_dir = os.path.relpath(abs_image_dir, current_dir)
            stored_paths = [os.path.relpath(p, current_dir) for p in final_data["paths"]]
        else:
            print("图片目录不在当前工作目录下，使用绝对路径保存。")
            stored_base_dir = abs_image_dir
            stored_paths = final_data["paths"]

        np.savez_compressed(
            self.index_file,
            features=np.array(final_data["features"]),
            paths=np.array(stored_paths),
            hashes=np.array(final_data["hashes"]),
            base_dir=np.array([stored_base_dir])
        )
        print("索引更新完成！")