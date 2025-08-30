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
        indexed_data, path_to_hash, hash_to_path = self._load_or_initialize_index()
        
        new, modified, deleted, moved = self._scan_and_compare(path_to_hash)

        summary = {
            "new": len(new),
            "modified": len(modified),
            "deleted": len(deleted),
            "moved": len(moved),
            "total": 0,
            "message": ""
        }

        if not any([new, modified, deleted, moved]):
            summary["total"] = len(indexed_data.get('paths', []))
            summary["message"] = "文件无变化，索引已是最新。"
            print(summary["message"])
            return summary

        print(f"\n检测结果: {len(new)} 新增, {len(modified)} 修改, {len(moved)} 移动/重命名, {len(deleted)} 删除。")
        final_data = self._process_changes(indexed_data, new, modified, deleted, moved)
        self._save_index(final_data)
        self._save_image_dir_to_config()

        summary["total"] = len(final_data.get('paths', []))
        summary["message"] = f"索引更新成功！"
        return summary

    def _load_or_initialize_index(self):
        """
        加载现有索引，并创建用于快速查找的辅助映射。
        返回 (索引数据, 路径->哈希映射, 哈希->路径映射)。
        """
        if not os.path.exists(self.index_file):
            print(f"未找到索引文件，将创建一个新索引。")
            return {"features": [], "paths": [], "hashes": []}, {}, {}

        print(f"正在加载现有索引 '{self.index_file}'...")
        try:
            data = np.load(self.index_file, allow_pickle=True)
            resolved_paths = [os.path.abspath(p) for p in data['paths']]
            hashes = list(data['hashes']) if 'hashes' in data else [None] * len(resolved_paths)

            indexed_data = {
                "features": list(data['features']),
                "paths": resolved_paths,
                "hashes": hashes
            }
            
            path_to_hash = {path: h for path, h in zip(resolved_paths, hashes)}
            hash_to_path = {h: path for path, h in zip(resolved_paths, hashes) if h is not None}
            
            return indexed_data, path_to_hash, hash_to_path
        except Exception as e:
            print(f"警告：加载索引文件失败，将创建一个新索引。错误: {e}")
            return {"features": [], "paths": [], "hashes": []}, {}, {}

    def _scan_and_compare(self, indexed_path_to_hash: dict):
        """
        扫描磁盘文件，并与索引数据比对，智能地检测出移动/重命名的文件。
        """
        # 1. 从磁盘获取当前所有文件的绝对路径
        current_paths_on_disk = set()
        for ext in IMAGE_EXTENSIONS:
            found_files = glob.glob(os.path.join(self.image_dir, '**', ext), recursive=True)
            current_paths_on_disk.update(os.path.abspath(path) for path in found_files)

        indexed_paths = set(indexed_path_to_hash.keys())

        # 2. 初步分类
        potentially_new_paths = current_paths_on_disk - indexed_paths
        potentially_deleted_paths = indexed_paths - current_paths_on_disk
        unchanged_or_modified_paths = current_paths_on_disk.intersection(indexed_paths)

        # 3. 找出内容被修改的文件
        modified = []
        for path in tqdm(unchanged_or_modified_paths, desc="检测文件修改"):
            current_hash = calculate_hash(path)
            if not current_hash: continue
            if current_hash != indexed_path_to_hash.get(path):
                modified.append(path)
        
        # 4. 智能检测移动/重命名的文件
        moved, truly_new = [], []
        deleted_hash_to_path = {
            indexed_path_to_hash[path]: path 
            for path in potentially_deleted_paths 
            if indexed_path_to_hash.get(path) is not None
        }

        for path in tqdm(potentially_new_paths, desc="检测新增/移动文件"):
            current_hash = calculate_hash(path)
            if not current_hash: continue

            if current_hash in deleted_hash_to_path:
                # 发现移动/重命名文件！
                old_path = deleted_hash_to_path[current_hash]
                moved.append({'old_path': old_path, 'new_path': path})
                del deleted_hash_to_path[current_hash] # 标记为已找到
            else:
                # 这是一个真正的新文件
                truly_new.append(path)
        
        # 5. 字典中剩下的是真正被删除的文件
        truly_deleted = list(deleted_hash_to_path.values())
        
        return truly_new, modified, truly_deleted, moved

    def _process_changes(self, indexed_data, new_paths, modified_paths, deleted_paths, moved_files):
        path_to_idx = {path: i for i, path in enumerate(indexed_data['paths'])}

        # 1. 处理移动文件：仅更新路径，不重新提取特征（核心优化点）
        if moved_files:
            print("正在为移动的文件更新路径...")
            for move_info in moved_files:
                old_path, new_path = move_info['old_path'], move_info['new_path']
                if old_path in path_to_idx:
                    idx = path_to_idx[old_path]
                    indexed_data['paths'][idx] = new_path
            # 路径已变，重建映射
            path_to_idx = {path: i for i, path in enumerate(indexed_data['paths'])}

        # 2. 准备一个干净的数据列表，移除待删除和待修改的旧条目
        final_data = {"features": [], "paths": [], "hashes": []}
        paths_to_remove = set(modified_paths + deleted_paths)
        for i, path in enumerate(indexed_data['paths']):
            if path not in paths_to_remove:
                final_data["features"].append(indexed_data["features"][i])
                final_data["paths"].append(path)
                final_data["hashes"].append(indexed_data["hashes"][i])

        # 3. 为真正新增和内容修改过的文件提取特征
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

                    if path_lower.endswith(('.webm', '.mp4', '.gif')):
                        cap = cv2.VideoCapture(path)
                        if cap.isOpened():
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            if total_frames > 0:
                                random_frame_index = random.randint(0, total_frames - 1)
                                cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
                                
                                ret, frame = cap.read()
                                if ret:
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    image = Image.fromarray(frame_rgb)
                            cap.release()
                        else:
                             print(f"\n警告：无法使用 OpenCV 打开文件 {path}，已跳过。")
                             continue
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