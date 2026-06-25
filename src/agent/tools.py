"""Agent 工具定义

5 个 LangChain Tool:
- search: 混合检索, 回答事实性问题
- compare: 多文档对比分析
- summarize: 主题摘要
- extract_table: 表格提取
- ask_clarification: 澄清问题
"""

from typing import List
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from src.config import SearchResult, settings
from src.rag.prompt import COMPARE_PROMPT, SUMMARIZE_PROMPT, TABLE_EXTRACTION_PROMPT


# 全局引用, 由 executor 初始化时注入
_retriever = None
_reranker = None


def set_retriever(retriever, reranker=None):
    """设置检索器 (由 executor 初始化时调用)"""
    global _retriever, _reranker
    _retriever = retriever
    _reranker = reranker


def _do_retrieve(query: str, top_k: int = 5) -> List[SearchResult]:
    """执行混合检索"""
    if _retriever is None:
        return []

    candidates = _retriever.search(query, top_k=top_k)
    if _reranker and candidates:
        candidates = _reranker.rerank(query, candidates)
    return candidates[:top_k]


def _context_from_results(results: List[SearchResult]) -> str:
    """将检索结果格式化为 LLM 上下文"""
    parts = []
    for i, r in enumerate(results, start=1):
        source = f"[{i}] 来源: {r.filename}"
        if r.page_number:
            source += f", 第{r.page_number}页"
        parts.append(f"{source}\n{r.content}")
    return "\n\n---\n\n".join(parts)


def _get_llm():
    """获取 LLM 实例"""
    llm_kwargs = {"model": settings.llm_model, "temperature": 0}
    if settings.effective_llm_base_url:
        llm_kwargs["base_url"] = settings.effective_llm_base_url
    return ChatOpenAI(openai_api_key=settings.effective_llm_api_key, **llm_kwargs)


@tool
def search(query: str) -> str:
    """
    在文档库中搜索特定事实或信息。
    适用于精确查找类问题, 如 "违约金比例是多少?"、"合同签署日期是什么?"

    Args:
        query: 搜索查询词, 应提取问题中的关键信息
    """
    results = _do_retrieve(query)
    if not results:
        return "未找到相关文档内容。"

    ctx = _context_from_results(results)
    llm = _get_llm()

    prompt = (
        f"根据以下检索到的文档内容回答问题。只根据文档内容回答, 不要编造。\n\n"
        f"## 文档内容\n{ctx}\n\n"
        f"## 问题\n{query}\n\n## 回答"
    )
    response = llm.invoke(prompt)
    return response.content


@tool
def compare(query: str) -> str:
    """
    比较多份文档中关于某个主题的不同说法。
    适用于对比分析类问题, 如 "两份合同中违约金条款有什么区别?"

    Args:
        query: 对比查询, 应明确要对哪些内容进行比较
    """
    results = _do_retrieve(query, top_k=8)
    if len(results) < 2:
        return "检索到的内容不足以进行对比。请尝试更宽泛的查询。"

    ctx = _context_from_results(results)
    llm = _get_llm()

    prompt = COMPARE_PROMPT.format(context=ctx, question=query)
    response = llm.invoke(prompt)
    return response.content


@tool
def summarize(query: str) -> str:
    """
    对某个主题在文档中的相关内容做摘要。
    适用于概括性问题, 如 "这份合同主要讲了什么?"

    Args:
        query: 摘要查询, 描述需要概括的主题
    """
    results = _do_retrieve(query, top_k=8)
    if not results:
        return "未找到相关文档内容。"

    ctx = _context_from_results(results)
    llm = _get_llm()

    prompt = SUMMARIZE_PROMPT.format(context=ctx, question=query)
    response = llm.invoke(prompt)
    return response.content


@tool
def extract_table(query: str) -> str:
    """
    从文档中提取表格数据并格式化输出。
    适用于表格类问题, 如 "列出产品规格参数表"

    Args:
        query: 表格提取查询, 指定要提取哪类表格信息
    """
    # 加强关键词, 帮助检索到表格相关内容
    enhanced_query = f"表格 数据 {query}"
    results = _do_retrieve(enhanced_query, top_k=8)
    if not results:
        return "未找到相关表格数据。"

    ctx = _context_from_results(results)
    llm = _get_llm()

    prompt = TABLE_EXTRACTION_PROMPT.format(context=ctx, question=query)
    response = llm.invoke(prompt)
    return response.content


@tool
def ask_clarification(clarification_question: str) -> str:
    """
    当用户问题不够明确时, 提出澄清性问题。

    Args:
        clarification_question: 向用户提出的澄清问题
    """
    return f"需要澄清: {clarification_question}"


# 导出工具列表
ALL_TOOLS = [search, compare, summarize, extract_table, ask_clarification]
