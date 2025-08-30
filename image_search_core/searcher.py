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
        data = np.load(self.index_file, allow_pickle=True)
        self.image_features = torch.from_numpy(data['features']).to(self.device)
        
        self.image_paths = [os.path.abspath(p) for p in data['paths']]
        self.path_to_idx = {path: i for i, path in enumerate(self.image_paths)}
        
        self.normalized_image_features = F.normalize(self.image_features, p=2, dim=-1)
        print(f"索引加载完毕，包含 {len(self.image_paths)} 张图片。")

    def search(self, query: str, top_k: int = 5, negative_query: str = None, similar_image_path: str = None, offset: int = 0):
        """
        根据文本和/或图片进行语义搜索。
        """
        if top_k <= 0 or not (query or similar_image_path):
            return []
            
        model, processor = self.model_loader.load()
        
        with torch.no_grad():
            # 1. 构建正面查询向量
            positive_vectors = []
            if query and query.strip():
                print(f"\n正在为查询 “{query}” 提取文本特征...")
                inputs = processor(text=[query], return_tensors="pt").to(self.device)
                text_features = model.get_text_features(**inputs)
                positive_vectors.append(text_features)

            if similar_image_path and similar_image_path.strip():
                image_idx = self.path_to_idx.get(similar_image_path)
                if image_idx is not None:
                    print(f"添加图片 “{os.path.basename(similar_image_path)}” 到搜索条件...")
                    image_features = self.image_features[image_idx].unsqueeze(0)
                    positive_vectors.append(image_features)
                else:
                    print(f"警告：相似性图片 '{similar_image_path}' 不在索引中，已忽略。")

            if not positive_vectors:
                return []

            # 将文本和图片向量平均，创建混合查询向量
            combined_positive_features = torch.mean(torch.cat(positive_vectors, dim=0), dim=0, keepdim=True)
            normalized_positive_features = F.normalize(combined_positive_features, p=2, dim=-1)
            
            # 2. 计算正面余弦相似度
            positive_scores = (normalized_positive_features @ self.normalized_image_features.T).squeeze()
            final_scores = positive_scores
            
            # 3. 如果有负面查询，处理它
            if negative_query and negative_query.strip():
                negative_keywords = [kw.strip() for kw in negative_query.split(',') if kw.strip()]
                if negative_keywords:
                    print(f"正在为排除项 {negative_keywords} 提取文本特征...")
                    neg_inputs = processor(text=negative_keywords, return_tensors="pt", padding=True).to(self.device)
                    neg_text_features = model.get_text_features(**neg_inputs)
                    avg_neg_features = neg_text_features.mean(dim=0, keepdim=True)
                    normalized_neg_text_features = F.normalize(avg_neg_features, p=2, dim=-1)
                    negative_scores = (normalized_neg_text_features @ self.normalized_image_features.T).squeeze()
                    final_scores = positive_scores - negative_scores
                
        # 4. 获取 Top K 结果（支持偏移量）
        total_results_to_fetch = min(top_k + offset, len(self.image_paths))
        if total_results_to_fetch <= offset:
            return []
            
        top_results = torch.topk(final_scores, k=total_results_to_fetch)
        
        scores = top_results.values.cpu().numpy()
        indices = top_results.indices.cpu().numpy()
        
        # 应用偏移量
        paginated_indices = indices[offset:]
        paginated_scores = scores[offset:]

        return [{"path": self.image_paths[idx], "score": float(score)} for idx, score in zip(paginated_indices, paginated_scores)]


    def search_by_image(self, image_path: str, top_k: int = 5, negative_query: str = None, offset: int = 0):
        """
        根据给定的图片，返回最相似的前 K 张图片，并应用排除关键词。
        """
        if top_k <= 0:
            return []

        query_idx = self.path_to_idx.get(image_path)
        if query_idx is None:
            raise ValueError(f"图片 '{os.path.basename(image_path)}' 不在索引中。")
        
        print(f"\n正在以图片 “{os.path.basename(image_path)}” 为基准查找相似项...")
        query_vector = self.normalized_image_features[query_idx].unsqueeze(0)
        
        # 计算正面相似度
        positive_scores = (query_vector @ self.normalized_image_features.T).squeeze()
        final_scores = positive_scores

        # 应用排除关键词
        if negative_query and negative_query.strip():
            model, processor = self.model_loader.load()
            with torch.no_grad():
                negative_keywords = [kw.strip() for kw in negative_query.split(',') if kw.strip()]
                if negative_keywords:
                    print(f"正在为排除项 {negative_keywords} 提取文本特征...")
                    neg_inputs = processor(text=negative_keywords, return_tensors="pt", padding=True).to(self.device)
                    neg_text_features = model.get_text_features(**neg_inputs)
                    avg_neg_features = neg_text_features.mean(dim=0, keepdim=True)
                    normalized_neg_features = F.normalize(avg_neg_features, p=2, dim=-1)
                    negative_scores = (normalized_neg_features @ self.normalized_image_features.T).squeeze()
                    final_scores = positive_scores - negative_scores

        # 获取足够多的结果以支持分页和排除自身
        # 请求 top_k + offset + 1 个结果以确保在排除自身后仍有足够的数据
        total_results_to_fetch = min(top_k + offset + 1, len(self.image_paths))
        top_results = torch.topk(final_scores, k=total_results_to_fetch)

        scores = top_results.values.cpu().numpy()
        indices = top_results.indices.cpu().numpy()

        # 排除自身
        all_results = []
        for idx, score in zip(indices, scores):
            if idx != query_idx:
                all_results.append({"path": self.image_paths[idx], "score": float(score)})
        
        # 应用偏移量和 top_k
        start = offset
        end = offset + top_k
        return all_results[start:end]