"""消融实验报告生成

对比 4 种检索策略: BM25-Only, Dense-Only, Hybrid, Hybrid+Reranker
"""

import time
import logging
from typing import List
from src.config import settings
from src.evaluation.dataset import load_eval_dataset, EvalExample
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.evaluation.answer_metrics import citation_accuracy

logger = logging.getLogger(__name__)


def _get_retriever():
    """初始化检索组件"""
    from src.vector_store.chroma_store import ChromaStore
    from src.retriever.bm25_retriever import BM25Retriever
    from src.retriever.dense_retriever import DenseRetriever
    from src.retriever.hybrid_retriever import HybridRetriever
    from src.retriever.reranker import Reranker

    store = ChromaStore()
    bm25 = BM25Retriever()
    dense = DenseRetriever(store)

    chunks = store.get_all_chunks()
    if chunks:
        bm25.build_index(chunks)

    hybrid = HybridRetriever(bm25, dense)
    reranker = Reranker()

    return bm25, dense, hybrid, reranker


def run_ablation(
    eval_path: str = "examples/qa_eval_set.jsonl",
    top_k: int = 5,
) -> dict:
    """运行消融实验"""
    examples = load_eval_dataset(eval_path)
    if not examples:
        logger.warning("评估集为空")
        return {}

    bm25, dense, hybrid, reranker = _get_retriever()

    strategies = {
        "BM25-Only": lambda q: bm25.search(q, top_k=top_k * 4),
        "Dense-Only": lambda q: dense.search(q, top_k=top_k * 4),
        "Hybrid (BM25+Dense+RRF)": lambda q: hybrid.search(q, top_k=top_k * 4),
        "Hybrid+Reranker": lambda q: reranker.rerank(
            q, hybrid.search(q, top_k=top_k * 4)
        )[:top_k],
    }

    results = {}

    for name, strategy_fn in strategies.items():
        print(f"\n{'='*50}")
        print(f"  评测: {name}")
        print(f"{'='*50}")

        total_recall_3 = 0.0
        total_recall_5 = 0.0
        total_mrr = 0.0
        total_ndcg = 0.0
        total_time = 0.0
        count = 0

        for ex in examples:
            if not ex.relevant_chunks:
                continue

            start = time.time()
            retrieved = strategy_fn(ex.question)
            elapsed = time.time() - start

            retrieved_ids = [r.chunk_id for r in retrieved]
            metrics = evaluate_retrieval(
                retrieved_ids, ex.relevant_chunks, k_values=[3, 5]
            )

            total_recall_3 += metrics.get("recall@3", 0)
            total_recall_5 += metrics.get("recall@5", 0)
            total_mrr += metrics.get("mrr", 0)
            total_ndcg += metrics.get("ndcg@5", 0)
            total_time += elapsed
            count += 1

        if count > 0:
            results[name] = {
                "Recall@3": round(total_recall_3 / count, 4),
                "Recall@5": round(total_recall_5 / count, 4),
                "MRR": round(total_mrr / count, 4),
                "NDCG@5": round(total_ndcg / count, 4),
                "Avg Latency (s)": round(total_time / count, 3),
            }

    return results


def print_ablation_report(results: dict) -> str:
    """生成可打印的消融实验报告"""
    if not results:
        return "无评估结果"

    headers = list(next(iter(results.values())).keys())
    strategy_names = list(results.keys())

    # Markdown 表格
    lines = ["# 消融实验报告\n"]
    lines.append("## 检索策略对比\n")

    # 表头
    header_line = "| 策略 | " + " | ".join(headers) + " |"
    sep_line = "|" + "|".join(" --- " for _ in range(len(headers) + 1)) + "|"
    lines.append(header_line)
    lines.append(sep_line)

    # 找最佳值用于标记
    best = {}
    for h in headers:
        is_latency = "Latency" in h
        vals = [(n, r[h]) for n, r in results.items()]
        if is_latency:
            best[h] = min(vals, key=lambda x: x[1])[0]
        else:
            best[h] = max(vals, key=lambda x: x[1])[0]

    for name, metrics in results.items():
        row = [name]
        for h in headers:
            val = metrics[h]
            marker = " **" if name == best.get(h) else ""
            row.append(f"{val}{marker}")
        lines.append("| " + " | ".join(row) + " |")

    # 结论
    lines.append("\n## 结论\n")
    lines.append("**最优策略标注为粗体。**\n")
    lines.append("预期结果: Hybrid+Reranker 在所有质量指标上最优，"
                 "但延迟略高于纯 Hybrid。")

    return "\n".join(lines)


def run_full_evaluation(eval_path: str = "examples/qa_eval_set.jsonl") -> dict:
    """运行完整评估"""
    results = run_ablation(eval_path)

    report = print_ablation_report(results)
    print(report)

    # 保存报告
    with open("ablation_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    return results


if __name__ == "__main__":
    run_full_evaluation()
