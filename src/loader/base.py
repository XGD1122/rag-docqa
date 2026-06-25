"""文档加载器抽象基类"""

from abc import ABC, abstractmethod
from src.config import Document


class DocumentLoader(ABC):
    """文档加载器抽象基类，所有加载器需实现 load 方法"""

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """加载文档文件，返回 Document 对象"""
        ...

    @staticmethod
    def supports(file_path: str) -> bool:
        """检查是否支持该文件类型"""
        return False
