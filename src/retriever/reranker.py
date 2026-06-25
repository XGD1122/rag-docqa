"""Re-ranker 精排器

使用 BGE-Reranker-v2-m3 cross-encoder 对候选进行细粒度排序。
"""

import logging
from typing import List
from src.config import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """BGE Cross-Encoder 精排器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self._model_name = model_name
        self._model = None
        self._loaded = False

    def _ensure_model(self):
        """延迟加载模型 (首次使用时下载)"""
        if self._loaded:
            return
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                self._model_name,
                use_fp16=True,
            )
            self._loaded = True
            logger.info(f"Reranker 模型已加载: {self._model_name}")
        except Exception as e:
            logger.warning(f"Reranker 模型加载失败: {e}. 将跳过精排步骤。")
            self._loaded = True  # 标记已尝试，避免反复重试

    def rerank(
        self, query: str, candidates: List[SearchResult]
    ) -> List[SearchResult]:
        """对候选结果精排"""
        if not candidates:
            return []

        self._ensure_model()

        if self._model is None:
            # 模型不可用，原样返回
            return candidates

        # 构建 [query, doc] 对
        pairs = [[query, c.content] for c in candidates]

        try:
            scores = self._model.compute_score(pairs, normalize=True)
        except Exception as e:
            logger.warning(f"Reranker 计算失败: {e}")
            return candidates

        # 处理单个结果的情况
        if isinstance(scores, float):
            scores = [scores]

        # 更新分数并重排序
        for i, candidate in enumerate(candidates):
            score = scores[i] if i < len(scores) else candidate.score
            candidate.source_scores["rerank"] = float(score)
            candidate.score = float(score)

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates
