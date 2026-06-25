"""检索评估指标

Recall@K, MRR (Mean Reciprocal Rank), NDCG@K
"""

import math
from typing import List


def recall_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """Recall@K: 前 K 个结果中命中了多少相关文档"""
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(retrieved_set & relevant_set)
    return hits / len(relevant_set)


def mrr(
    retrieved_ids: List[str], relevant_ids: List[str]
) -> float:
    """MRR (Mean Reciprocal Rank): 第一个相关结果的倒数排名"""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """NDCG@K: 归一化折损累计增益

    相关文档 ID 匹配则 gain=1, 否则 gain=0
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        gain = 1.0 if doc_id in relevant_set else 0.0
        dcg += gain / math.log2(i + 1)

    # IDCG (理想排序: 所有相关文档排在最前面)
    idcg = 0.0
    for i in range(1, min(len(relevant_ids), k) + 1):
        idcg += 1.0 / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_retrieval(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k_values: List[int] | None = None,
) -> dict:
    """批量计算所有检索指标"""
    if k_values is None:
        k_values = [3, 5, 10]

    results = {}
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

    results["mrr"] = mrr(retrieved_ids, relevant_ids)
    return results
