"""回答质量评估指标

Citation Accuracy: 引用命中相关 chunk 的比例
Faithfulness: LLM 判断回答是否忠实于文档 (基于 LLM 评估)
"""

from typing import List
from src.config import Citation, SearchResult, settings


def citation_accuracy(
    citations: List[Citation], relevant_chunk_ids: List[str]
) -> float:
    """计算引用准确率: 有引用的声明中有多少命中了相关 chunk"""
    if not citations:
        return 0.0

    cited_count = 0
    hit_count = 0
    relevant_set = set(relevant_chunk_ids)

    for c in citations:
        if c.source_chunk_id is not None:
            cited_count += 1
            if c.source_chunk_id in relevant_set:
                hit_count += 1

    if cited_count == 0:
        return 0.0

    return hit_count / cited_count


def faithfulness(
    answer: str, evidence: List[SearchResult]
) -> dict:
    """LLM 评估回答忠实度

    Returns:
        {"score": float (0-1), "reason": str}
    """
    if not evidence:
        return {"score": 0.0, "reason": "无证据可验证"}

    try:
        from langchain_openai import ChatOpenAI
        from src.rag.prompt import CLAIM_EXTRACTION_PROMPT

        llm_kwargs = {"model": settings.llm_model, "temperature": 0}
        if settings.effective_llm_base_url:
            llm_kwargs["base_url"] = settings.effective_llm_base_url
        llm = ChatOpenAI(
            openai_api_key=settings.effective_llm_api_key,
            **llm_kwargs,
        )

        # 提取声明
        prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        response = llm.invoke(prompt)

        # 简化: 用证据覆盖度评估
        evidence_text = "\n\n".join(e.content[:500] for e in evidence[:5])

        eval_prompt = f"""评估以下回答是否忠实于提供的文档证据。

## 回答
{answer}

## 文档证据
{evidence_text[:3000]}

请以 JSON 格式返回评估结果:
{{"score": 0.0-1.0, "reason": "评估理由"}}

其中 score 表示回答中被证据支持的比例 (0=完全不支持, 1=完全支持)。
"""
        response = llm.invoke(eval_prompt)

        import json
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        result = json.loads(content)
        return {"score": float(result.get("score", 0)), "reason": result.get("reason", "")}

    except Exception as e:
        return {"score": -1.0, "reason": f"评估失败: {str(e)}"}
