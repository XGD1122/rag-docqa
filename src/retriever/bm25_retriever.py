"""BM25 关键词检索器

使用 rank_bm25 库实现纯 Python BM25 检索。
索引仅存于内存，从 ChromaDB 重建。
"""

from typing import List, Tuple
from rank_bm25 import BM25Okapi
from src.config import Chunk, SearchResult, settings


class BM25Retriever:
    """BM25 关键词检索器 (内存索引)"""

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._chunks: List[Chunk] = []
        self._corpus: List[List[str]] = []  # tokenized corpus

    def build_index(self, chunks: List[Chunk]) -> None:
        """从 chunk 列表构建 BM25 内存索引"""
        self._chunks = chunks
        self._corpus = [self._tokenize(c.content) for c in chunks]
        if self._corpus:
            self._index = BM25Okapi(self._corpus)
        else:
            self._index = None

    def search(
        self, query: str, top_k: int | None = None
    ) -> List[SearchResult]:
        """BM25 检索"""
        k = top_k or settings.top_k

        if self._index is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # 按分数排序取 top_k
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = indexed_scores[:k]

        # 归一化分数到 0-1
        max_score = max(scores) if len(scores) > 0 else 1.0

        results = []
        for idx, raw_score in top_indices:
            if raw_score <= 0:
                continue
            chunk = self._chunks[idx]
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=raw_score / max_score if max_score > 0 else 0.0,
                filename=chunk.filename,
                page_number=chunk.page_number,
                source_scores={"bm25": raw_score / max_score if max_score > 0 else 0.0},
            ))

        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词: 按空格和标点切分"""
        import re
        # 保留中文字符和英文单词
        tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower())
        return tokens if tokens else text.lower().split()
