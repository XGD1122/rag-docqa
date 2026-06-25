"""RAG Pipeline

串联: 检索 → 生成 → 引用匹配 → 反思自检
"""

import logging
from typing import List
from langchain_openai import ChatOpenAI
from src.config import SearchResult, Citation, ReflectionResult, settings
from src.rag.prompt import QA_PROMPT
from src.rag.citation import CitationMatcher
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.reranker import Reranker

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG 完整流水线"""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker | None = None,
        citation_matcher: CitationMatcher | None = None,
    ):
        self._retriever = retriever
        self._reranker = reranker
        self._citation = citation_matcher or CitationMatcher()

        llm_kwargs = {
            "model": settings.llm_model,
            "temperature": 0,
        }
        if settings.effective_llm_base_url:
            llm_kwargs["base_url"] = settings.effective_llm_base_url
        self._llm = ChatOpenAI(
            openai_api_key=settings.effective_llm_api_key,
            **llm_kwargs,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> List[SearchResult]:
        """检索阶段: Hybrid + Reranker"""
        k = top_k or settings.top_k

        # 混合检索获取候选 (Hybrid 内部已做 RRF 融合)
        candidates = self._retriever.search(query, top_k=k)

        # Reranker 精排
        if self._reranker and candidates:
            candidates = self._reranker.rerank(query, candidates)

        return candidates[:k]

    def generate(self, query: str, context: str) -> str:
        """LLM 生成回答"""
        prompt = QA_PROMPT.format(context=context, question=query)
        response = self._llm.invoke(prompt)
        return response.content

    def cite(
        self, answer: str, retrieved_chunks: List[SearchResult]
    ) -> List[Citation]:
        """后置引用匹配"""
        return self._citation.match_claims(answer, retrieved_chunks)

    def query(
        self, question: str, run_reflection: bool = True
    ) -> dict:
        """执行完整 RAG 流程

        Returns:
            {
                "answer": str,
                "citations": List[Citation],
                "retrieved_chunks": List[SearchResult],
                "reflection": ReflectionResult | None,
            }
        """
        # 1. 检索
        retrieved = self.retrieve(question)
        if not retrieved:
            return {
                "answer": "未在文档中找到相关信息。",
                "citations": [],
                "retrieved_chunks": [],
                "reflection": None,
            }

        # 2. 构建上下文
        context_parts = []
        for i, r in enumerate(retrieved, start=1):
            source = f"[来源: {r.filename}"
            if r.page_number:
                source += f", 第{r.page_number}页"
            source += "]"
            context_parts.append(f"{source}\n{r.content}")
        context = "\n\n---\n\n".join(context_parts)

        # 3. 生成回答
        answer = self.generate(question, context)

        # 4. 引用匹配
        citations = self.cite(answer, retrieved)

        # 5. 反思自检 (可选)
        reflection = None
        if run_reflection:
            try:
                from src.agent.reflection import ReflectionChecker
                checker = ReflectionChecker()
                reflection = checker.check(answer, retrieved)
            except Exception as e:
                logger.warning(f"反思检查失败: {e}")

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved,
            "reflection": reflection,
        }
