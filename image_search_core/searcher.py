# image_search_core/searcher.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from .model_loader import ModelLoader
from .config import DEFAULT_INDEX_FILE

class ImageSearcher:
    """
    封装了加载索引和执行语义搜索的所有逻辑。
    初始化时会预加载索引数据到内存，以实现快速搜索。
    """
    def __init__(self, index_file: str = DEFAULT_INDEX_FILE, model_path: str = None):
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"索引文件 '{index_file}' 不存在。请先运行索引程序。")
        self.index_file = index_file
        self.model_loader = ModelLoader(model_path)
        self.device = self.model_loader.device
        
        print(f"正在从 '{self.index_file}' 加载索引数据...")
        data = np.load(self.index_file)
        self.image_features = torch.from_numpy(data['features']).to(self.device)
        self.image_paths = data['paths']
        # 预先进行归一化，加速后续计算
        self.normalized_image_features = F.normalize(self.image_features, p=2, dim=-1)
        print(f"索引加载完毕，包含 {len(self.image_paths)} 张图片。")

    def search(self, query: str, top_k: int = 5):
        """
        根据中文文本查询，返回最相似的前 K 张图片。
        返回: 一个字典列表，例如 [{'path': '...', 'score': 0.85}, ...]
        """
        if top_k <= 0:
            return []
            
        model, processor = self.model_loader.load()
        
        print(f"\n正在为查询 “{query}” 提取文本特征...")
        with torch.no_grad():
            inputs = processor(text=[query], return_tensors="pt").to(self.device)
            text_features = model.get_text_features(**inputs)
            normalized_text_features = F.normalize(text_features, p=2, dim=-1)

        # 计算余弦相似度
        similarity_scores = (normalized_text_features @ self.normalized_image_features.T).squeeze()
        
        # 获取 Top K 结果
        top_k = min(top_k, len(self.image_paths))
        top_results = torch.topk(similarity_scores, k=top_k)
        
        scores = top_results.values.cpu().numpy()
        indices = top_results.indices.cpu().numpy()
        
        return [{"path": self.image_paths[idx], "score": float(score)} for idx, score in zip(indices, scores)]