# image_search_core/model_loader.py
import json
from .config import CONFIG_FILE
from .utils import get_device

class ModelLoader:
    """
    一个专门用于加载和管理 Chinese-CLIP 模型和处理器的类。
    实现了懒加载，只有在第一次需要时才会真正加载模型到内存。
    """
    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or self._get_path_from_config()
        self.device = device or get_device()
        self.model = None
        self.processor = None

    def _get_path_from_config(self) -> str:
        """从配置文件读取模型路径。"""
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f).get("model_path")
        except FileNotFoundError:
            raise RuntimeError(f"配置文件 '{CONFIG_FILE}' 未找到。请先下载模型到当前文件夹，并新建此配置文件，键 model_path 的值为文件夹的名称。")

    def load(self):
        """
        加载模型和处理器（如果尚未加载）。
        返回: (model, processor) 元组。
        """
        if self.model is None or self.processor is None:
            print(f"正在从 '{self.model_path}' 加载模型到设备 '{self.device}'...")
            try:
                # 动态导入以避免在所有地方都依赖 transformers
                from transformers import ChineseCLIPModel, ChineseCLIPProcessor
                self.model = ChineseCLIPModel.from_pretrained(self.model_path).to(self.device)
                self.processor = ChineseCLIPProcessor.from_pretrained(self.model_path, use_fast=True)
                self.model.eval() # 始终设置为评估模式
                print("模型加载成功。")
            except (OSError, ImportError) as e:
                raise RuntimeError(f"加载模型失败: {e}")
        return self.model, self.processor