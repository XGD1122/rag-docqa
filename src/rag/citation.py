"""后置匹配引用

在 LLM 生成回答后, 逐句计算 embedding 并与检索到的 chunk 做余弦相似度匹配,
找到每个声明的来源。
"""

import re
import logging
from typing import List
import numpy as np
from langchain_openai import OpenAIEmbeddings
from src.config import Citation, SearchResult, settings

logger = logging.getLogger(__name__)


class CitationMatcher:
    """后置引用匹配器"""

    def __init__(self, similarity_threshold: float = 0.6):
        self._threshold = similarity_threshold

        embed_kwargs = {"model": settings.embedding_model}
        if settings.effective_embedding_base_url:
            embed_kwargs["base_url"] = settings.effective_embedding_base_url
        self._embeddings = OpenAIEmbeddings(
            openai_api_key=settings.effective_embedding_api_key,
            check_embedding_ctx_length=False,
            **embed_kwargs,
        )

    def match_claims(
        self, answer: str, retrieved_chunks: List[SearchResult]
    ) -> List[Citation]:
        """逐句匹配来源"""
        if not answer.strip():
            return []

        # 步骤 1: 拆分句子
        sentences = self._split_sentences(answer)

        if not retrieved_chunks:
            return [
                Citation(sentence=s, similarity=0.0)
                for s in sentences
            ]

        # 步骤 2: 计算 sentence embeddings
        sentence_embeddings = self._embeddings.embed_documents(sentences)

        # 步骤 3: 计算 chunk embeddings
        chunk_texts = [c.content for c in retrieved_chunks]
        chunk_embeddings = self._embeddings.embed_documents(chunk_texts)

        # 步骤 4: 逐句匹配 (余弦相似度)
        citations = []
        for i, sentence in enumerate(sentences):
            sent_emb = np.array(sentence_embeddings[i])

            best_similarity = -1.0
            best_chunk: SearchResult | None = None

            for j, chunk in enumerate(retrieved_chunks):
                chunk_emb = np.array(chunk_embeddings[j])
                # 余弦相似度
                sim = np.dot(sent_emb, chunk_emb) / (
                    np.linalg.norm(sent_emb) * np.linalg.norm(chunk_emb) + 1e-10
                )

                if sim > best_similarity:
                    best_similarity = sim
                    best_chunk = chunk

            if best_similarity >= self._threshold and best_chunk is not None:
                citations.append(Citation(
                    sentence=sentence,
                    source_chunk_id=best_chunk.chunk_id,
                    source_text=best_chunk.content[:300],
                    similarity=round(best_similarity, 4),
                    filename=best_chunk.filename,
                    page_number=best_chunk.page_number,
                ))
            else:
                citations.append(Citation(
                    sentence=sentence,
                    similarity=round(best_similarity, 4) if best_similarity > 0 else 0.0,
                ))

        return citations

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """按句号、换行等拆分句子 (支持中英文)"""
        # 先按换行拆分
        parts = text.split("\n")
        sentences = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # 再按中文句号、问号、感叹号拆分
            sub_parts = re.split(r"(?<=[。！？\.\!\?])", part)
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    sentences.append(sp)
        return sentences
