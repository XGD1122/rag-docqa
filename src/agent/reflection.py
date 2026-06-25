"""LLM 自检反思模块

Step 1: LLM 提取回答中的关键事实声明
Step 2: 逐声明与 evidence 比对 (LLM 判断)
Step 3: 生成补充检索 query
Step 4: 返回状态
"""

import json
import logging
from typing import List
from langchain_openai import ChatOpenAI
from src.config import SearchResult, ReflectionResult, settings
from src.rag.prompt import (
    CLAIM_EXTRACTION_PROMPT,
    CLAIM_VERIFICATION_PROMPT,
    FOLLOWUP_QUERY_PROMPT,
)

logger = logging.getLogger(__name__)


class ReflectionChecker:
    """LLM 自检器: 验证回答是否被文档证据支持"""

    def __init__(self):
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

    def check(
        self, answer: str, evidence: List[SearchResult]
    ) -> ReflectionResult:
        """执行自检"""
        if not answer.strip():
            return ReflectionResult(status="pass")

        # Step 1: 提取声明
        claims = self._extract_claims(answer)
        if not claims:
            return ReflectionResult(status="pass")

        # Step 2: 逐声明与证据比对
        evidence_text = self._format_evidence(evidence)
        unsupported = []
        supported_claims = []

        for claim in claims:
            is_supported = self._verify_claim(claim, evidence_text)
            if is_supported:
                supported_claims.append(claim)
            else:
                unsupported.append(claim)

        # Step 3: 判断状态
        if not unsupported:
            return ReflectionResult(
                status="pass",
                claims=claims,
            )

        # 尝试生成补充查询
        followup_queries = self._generate_followup_queries(unsupported)

        if followup_queries:
            return ReflectionResult(
                status="retry",
                claims=claims,
                unsupported_claims=unsupported,
                followup_queries=followup_queries,
            )
        else:
            return ReflectionResult(
                status="ask_clarification",
                claims=claims,
                unsupported_claims=unsupported,
            )

    def _extract_claims(self, answer: str) -> List[str]:
        """Step 1: 从回答中提取关键事实声明"""
        try:
            prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
            response = self._llm.invoke(prompt)
            result = self._parse_json(response.content)
            return result.get("claims", [])
        except Exception as e:
            logger.warning(f"声明提取失败: {e}")
            return []

    def _verify_claim(self, claim: str, evidence_text: str) -> bool:
        """Step 2: LLM 判断声明是否被证据支持"""
        try:
            prompt = CLAIM_VERIFICATION_PROMPT.format(
                claim=claim, evidence=evidence_text[:4000]
            )
            response = self._llm.invoke(prompt)
            result = self._parse_json(response.content)
            return result.get("supported", False)
        except Exception as e:
            logger.warning(f"声明验证失败: {e}")
            return True  # 失败时默认信任回答

    def _generate_followup_queries(self, unsupported: List[str]) -> List[str]:
        """Step 3: 为未支持的声明生成补充检索查询"""
        try:
            claims_text = "\n".join(f"- {c}" for c in unsupported)
            prompt = FOLLOWUP_QUERY_PROMPT.format(unsupported_claims=claims_text)
            response = self._llm.invoke(prompt)
            result = self._parse_json(response.content)
            return result.get("queries", [])
        except Exception as e:
            logger.warning(f"补充查询生成失败: {e}")
            return []

    @staticmethod
    def _format_evidence(evidence: List[SearchResult]) -> str:
        """格式化证据文本"""
        parts = []
        for i, e in enumerate(evidence, start=1):
            source = f"[证据{i}] {e.filename}"
            if e.page_number:
                source += f" 第{e.page_number}页"
            parts.append(f"{source}\n{e.content}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_json(text: str) -> dict:
        """解析 LLM 返回的 JSON"""
        # 去除 markdown 代码块标记
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)
