"""混合检索器

融合 BM25 (关键词) + Dense (语义) 结果，使用 RRF (Reciprocal Rank Fusion) 算法。
"""

from typing import List
from src.config import SearchResult, settings
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.dense_retriever import DenseRetriever


class HybridRetriever:
    """混合检索器: BM25 + Dense + RRF 融合"""

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int | None = None,
    ):
        self._bm25 = bm25
        self._dense = dense
        self._rrf_k = rrf_k or settings.rrf_k

    def search(
        self, query: str, top_k: int | None = None
    ) -> List[SearchResult]:
        """执行混合检索"""
        k = top_k or settings.top_k
        fetch_k = k * 4  # 每种检索拉取更多候选

        # 第一阶段: 分别检索
        bm25_results = self._bm25.search(query, top_k=fetch_k)
        dense_results = self._dense.search(query, top_k=fetch_k)

        # 第二阶段: RRF 融合
        merged = self._rrf_fusion(bm25_results, dense_results, k)

        return merged

    def _rrf_fusion(
        self,
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """RRF 融合算法

        score(doc) = Σ 1 / (k + rank_i(doc))
        其中 k=60, rank_i(doc) 是文档在第 i 个检索器中的排名 (1-based)
        """
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, SearchResult] = {}

        # BM25 排名
        for rank, result in enumerate(bm25_results, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + \
                1.0 / (self._rrf_k + rank)
            if result.chunk_id not in chunk_map:
                chunk_map[result.chunk_id] = result
            # 更新 source_scores
            chunk_map[result.chunk_id].source_scores["bm25"] = result.score

        # Dense 排名
        for rank, result in enumerate(dense_results, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + \
                1.0 / (self._rrf_k + rank)
            if result.chunk_id not in chunk_map:
                chunk_map[result.chunk_id] = result
            chunk_map[result.chunk_id].source_scores["dense"] = result.score

        # 按 RRF 分数排序
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 归一化 RRF 分数并构建结果
        max_rrf = sorted_ids[0][1] if sorted_ids else 1.0
        results = []
        for chunk_id, rrf_score in sorted_ids[:top_k]:
            result = chunk_map[chunk_id]
            result.score = rrf_score / max_rrf if max_rrf > 0 else 0.0
            result.source_scores["rrf"] = result.score
            results.append(result)

        return results
