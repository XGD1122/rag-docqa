"""Dense 向量检索器

对 ChromaStore 的薄封装，提供语义向量检索。
"""

from typing import List
from src.config import SearchResult, settings
from src.vector_store.chroma_store import ChromaStore


class DenseRetriever:
    """向量语义检索器"""

    def __init__(self, chroma_store: ChromaStore):
        self._store = chroma_store

    def search(
        self, query: str, top_k: int | None = None
    ) -> List[SearchResult]:
        """执行语义检索"""
        k = top_k or settings.top_k
        return self._store.search(query, top_k=k)
