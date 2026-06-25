"""评估数据集加载"""

import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalExample:
    """单条评估样本"""
    question: str
    answer: str
    relevant_chunks: List[str] = field(default_factory=list)
    question_type: str = "exact_qa"
    documents: List[str] = field(default_factory=list)


def load_eval_dataset(path: str) -> List[EvalExample]:
    """从 JSONL 文件加载评估数据集"""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(EvalExample(
                question=data["question"],
                answer=data["answer"],
                relevant_chunks=data.get("relevant_chunks", []),
                question_type=data.get("question_type", "exact_qa"),
                documents=data.get("documents", []),
            ))
    return examples
