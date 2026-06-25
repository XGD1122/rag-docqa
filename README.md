# SmartRAG-Agent — Agentic RAG 智能文档系统

> 求职定位: **主项目**。展示 RAG 架构、Agent 工具调用、混合检索算法、引用追踪、自我反思、评估体系。
>
> 面试一句话: "实现了一个 Agentic RAG 系统，用 LangChain AgentExecutor 做工具路由，BM25+Dense+RRF+Reranker 做混合检索，后置匹配做引用、LLM 自检做反思，并且构建了评估集做消融实验。"

---

## 1. 你要做什么

做一个**能处理本地文档的智能问答系统**，带 Web UI。它不是简单的"向量检索 + LLM 回答"，核心差异:

| 普通 RAG | 这个项目 |
|----------|---------|
| 一个 retriever 搜一下 | BM25 + Dense + RRF 融合 + Reranker 四层检索 |
| LLM 直接回答 | Agent 先判断问题类型，选合适的工具 |
| 回答无来源 | 每个声明绑定文档名+页码+原文片段 |
| 答完就完了 | LLM 自检回答，证据不足自动补充检索 |
| 没法评估 | 有评估集 + Recall/MRR/NDCG + 消融实验 |

---

## 2. 技术栈

| 层 | 选型 | 为什么 |
|----|------|--------|
| LLM | OpenAI GPT-4o-mini | 便宜、稳定、function calling 成熟 |
| Embedding | OpenAI text-embedding-3-small | 1536 维，性价比高 |
| 向量库 | ChromaDB | 本地持久化、零配置 |
| Agent 框架 | LangChain AgentExecutor + OpenAI functions | 标准方案，`return_intermediate_steps` 可拿执行轨迹 |
| BM25 | rank_bm25 | 纯 Python，零外部依赖 |
| Reranker | BGE-Reranker-v2-m3 (或 sentence-transformers cross-encoder) | 开源精排模型 |
| 文档解析 | PyMuPDF (PDF) + python-docx (Word) + BeautifulSoup (网页) | 覆盖主流格式 |
| UI | Streamlit | 纯 Python，快速出界面 |
| 配置 | pydantic-settings | 类型安全，从 .env 加载 |

---

## 3. 系统架构

```text
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ 文档管理  │  │   问答区      │  │  Agent Trace +    │  │
│  │ 上传/列表 │  │  聊天+引用    │  │  Source Viewer    │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Agent Layer                            │
│  LangChain AgentExecutor + OpenAI function calling       │
│  Tools: search / compare / summarize /                   │
│         extract_table / ask_clarification                │
└─────────────────────────────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Retriever   │ │  Generator   │ │  Reflection  │
│  BM25+Dense  │ │  LLM +       │ │  Claim提取   │
│  +RRF+Rerank │ │  Citation    │ │  +证据比对   │
└──────────────┘ └──────────────┘ └──────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│         ChromaDB (持久化)              │
│  text-embedding-3-small embeddings   │
│  + document metadata                 │
└──────────────────────────────────────┘
```

---

## 4. 项目文件结构

每个文件做什么，按实现顺序排列:

```text
smartrag-agent/
├── src/
│   ├── config.py                  # [1] pydantic-settings, 读取 .env
│   ├── loader/
│   │   ├── base.py                # [2] DocumentLoader 抽象基类
│   │   ├── pdf_loader.py          # [2] PDF → 文本 + 页码元数据
│   │   ├── docx_loader.py         # [2] Word → 文本
│   │   └── text_loader.py         # [2] TXT/Markdown → 文本
│   ├── splitter/
│   │   └── semantic_chunker.py   # [3] 按标题+段落切分, 保留页码
│   ├── vector_store/
│   │   └── chroma_store.py       # [4] ChromaDB CRUD: add/delete/search/list
│   ├── retriever/
│   │   ├── bm25_retriever.py     # [5] rank_bm25 关键词检索
│   │   ├── dense_retriever.py    # [5] ChromaDB 向量检索
│   │   ├── hybrid_retriever.py   # [6] 融合 BM25 + Dense + RRF
│   │   └── reranker.py           # [6] BGE cross-encoder 精排
│   ├── agent/
│   │   ├── tools.py              # [7] 5个 LangChain Tool 定义
│   │   ├── executor.py           # [7] AgentExecutor 封装
│   │   ├── trace.py              # [7] 执行轨迹采集
│   │   └── reflection.py         # [8] LLM 自检: 提取声明→证据比对
│   ├── rag/
│   │   ├── pipeline.py           # [9] 串联检索→生成→引用→反思
│   │   ├── citation.py           # [9] 后置匹配: 逐句找来源
│   │   └── prompt.py             # [7] 所有 prompt 模板集中管理
│   └── evaluation/
│       ├── dataset.py            # [10] 加载 qa_eval_set.jsonl
│       ├── retrieval_metrics.py  # [10] Recall@K, MRR, NDCG
│       ├── answer_metrics.py     # [10] Citation Accuracy, Faithfulness
│       └── report.py             # [10] 消融实验报告生成
├── ui/
│   └── app.py                     # [11] Streamlit 三栏布局
├── examples/
│   ├── sample_docs/               # 测试用 PDF/DOCX/TXT
│   └── qa_eval_set.jsonl          # [10] 30-50条标注 QA
├── tests/                         # 单元测试
├── README.md                      # 本文件
├── requirements.txt
└── .env.example                   # OPENAI_API_KEY=xxx
```

---

## 5. 核心算法详解

### 5.1 HybridRetriever — 混合检索

```
输入 query
    │
    ├─→ BM25 (rank_bm25, 关键词匹配) → top_k*4 结果
    │
    └─→ Dense (ChromaDB, 语义匹配)   → top_k*4 结果
              │
              ▼
         RRF 融合 (k=60)
         score(doc) = Σ 1/(k+rank_i(doc))
              │
              ▼
         BGE-Reranker (cross-encoder 逐对打分)
              │
              ▼
         Top-K 最终结果
```

**为什么用 RRF 而不是加权求和**: BM25 和 dense 的分数不在同一量级，直接加权很难调参。RRF 用排名做融合，参数只需要一个 k 值。

### 5.2 BM25 索引生命周期

```
文档上传
  → ChromaDB.add(chunks, embeddings)     # 持久化到磁盘
  → BM25Builder.rebuild(all_chunks)      # 从 ChromaDB 读全量 chunk，重建内存索引
  → 后续检索用内存中 BM25 对象

文档删除
  → ChromaDB.delete(ids)
  → BM25Builder.rebuild(all_chunks)      # 同上
```

- 用 `rank_bm25` 库，纯 Python。
- BM25 **只存内存，不持久化**。ChromaDB 是唯一数据源。
- 重建很快（几千个 chunk 秒级完成）。

### 5.3 Citation — 后置匹配引用

```
LLM 生成回答（不含引用标记）
        │
        ▼
按句号/换行 split → [声明1, 声明2, 声明3...]
        │
        ▼
逐声明计算 embedding → 与所有检索到的 chunk 做余弦相似度
        │
        ▼
每个声明取最高相似度 chunk:
  相似度 ≥ 0.6 → [来源: 文档名, p.3, "原文片段..."]
  相似度 < 0.6 → [未找到直接来源]
```

**为什么不用 LLM 直接生成 `[1]` `[2]` 标记**: LLM 在长回答中经常编造引用编号，指向错误位置。后置匹配基于 embedding 计算，引用来源可验证。

### 5.4 ReflectionChecker — LLM 自检

```python
Step 1: LLM 提取回答中的关键事实声明
  answer = "违约金为5%，逾期超过30天触发。"
  claims = ["违约金比例为5%", "逾期超过30天触发"]

Step 2: 逐声明与 evidence 比对
  claim="违约金比例为5%" + evidence=["合同第3条: ...违约金为总金额5%..."]
  → LLM 判断: supported=True, source=chunk_3

  claim="逾期超过30天触发" + evidence=[...]
  → LLM 判断: supported=False  (evidence 中没有相关内容)

Step 3: 生成补充检索 query
  unsupported = ["逾期超过30天触发"]
  → followup_queries = ["逾期触发条件", "违约超过多少天触发罚则"]

Step 4: 返回
  all_supported  → pass
  有补充query    → retry (用新 query 重新检索)
  无法生成query  → ask_clarification
```

**为什么用 LLM 判断而非 embedding 相似度**: 同一事实可以用完全不同措辞表达。例如"违约金5%"和"按总金额百分之五收取罚金"——embedding 可能匹配不上，但 LLM 能正确判断。

### 5.5 Agent 路由决策

使用 LangChain AgentExecutor + OpenAI function calling:

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent

tools = [search_tool, compare_tool, summarize_tool,
         extract_table_tool, ask_clarification_tool]

agent = create_openai_functions_agent(ChatOpenAI(model="gpt-4o-mini"), tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)

result = executor.invoke({"input": user_question})
# result["intermediate_steps"] 包含 (AgentAction, observation) 轨迹
```

---

## 6. 实现顺序

按依赖关系分阶段:

### 阶段 A: 基础设施 (先做)
1. `config.py` — Settings dataclass, 从 .env 读 OPENAI_API_KEY
2. `loader/` — 4 个文件, 实现 PDF/DOCX/TXT/Markdown 加载
3. `splitter/semantic_chunker.py` — 按标题/段落切分
4. `vector_store/chroma_store.py` — ChromaDB 增删查
5. `.env.example` + `requirements.txt`

**验证**: 传一个 PDF, 能看到 chunk 列表 + 向量搜索结果。

### 阶段 B: 检索链路
5. `retriever/bm25_retriever.py` — rank_bm25 封装
6. `retriever/dense_retriever.py` — ChromaDB 向量检索封装
7. `retriever/hybrid_retriever.py` — RRF 融合
8. `retriever/reranker.py` — BGE cross-encoder

**验证**: 手动测试几个 query, 看四种策略的结果排序差异。

### 阶段 C: RAG 核心
9. `rag/prompt.py` — QA prompt 模板
10. `rag/citation.py` — 后置匹配引用
11. `rag/pipeline.py` — 检索 → 生成 → 引用 → 反思 串联
12. `agent/tools.py` — 5 个 Tool 定义
13. `agent/executor.py` — AgentExecutor 封装
14. `agent/trace.py` — 执行轨迹采集
15. `agent/reflection.py` — 自检模块

**验证**: 上传 2-3 份文档, 问不同场景的问题, 检查回答质量和引用准确性。

### 阶段 D: UI
16. `ui/app.py` — 三栏布局:
   - 左侧: 文档上传 + 已上传列表 + 删除按钮
   - 中间: 聊天界面 (历史消息 + 输入框 + 引用展示)
   - 右侧: Agent Trace (Thought→Action→Observation) + Source Viewer (点击引用查看原文)

### 阶段 E: 评估
17. `examples/qa_eval_set.jsonl` — 标注 30-50 条 QA
18. `evaluation/dataset.py` — 加载评估集
19. `evaluation/retrieval_metrics.py` — Recall@3/5, MRR, NDCG@5
20. `evaluation/answer_metrics.py` — Citation Accuracy, Faithfulness
21. `evaluation/report.py` — 消融实验: BM25/Dense/Hybrid/Hybrid+Reranker

**验证**: 跑消融实验, 确认 Hybrid+Reranker 全面最优。

---

## 7. 快速开始

```bash
cd smartrag-agent

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env, 填入 OPENAI_API_KEY=sk-xxx

# 3. 启动
streamlit run ui/app.py

# 4. 运行评估 (可选)
python -m src.evaluation.report
```

---

## 8. 评估集构建指南 (`qa_eval_set.jsonl`)

每行一条 JSON, 共 30-50 条:

```json
{
  "question": "违约金比例是多少?",
  "answer": "违约金比例为合同总金额的5%。",
  "relevant_chunks": ["contract_v1_chunk_3", "contract_v1_chunk_5"],
  "question_type": "exact_qa",
  "documents": ["contract_v1.pdf"]
}
```

覆盖 5 种场景各 6-10 条:
- **exact_qa**: 精确事实查找
- **compare**: 多文档对比
- **summary**: 摘要类
- **table**: 表格信息提取
- **missing**: 故意问文档中没有的信息 (测试"证据不足"处理)

标注方式: 手工标注 `relevant_chunks` (标记哪些 chunk 包含正确答案)。

---

## 9. 面试展示要点

1. **先展示 UI**: 打开 Streamlit, 上传文档, 问问题, 展示完整的引用和 trace。
2. **再讲架构**: 在白板/口头描述四层检索 + Agent + Reflection。
3. **最后讲评估**: 展示消融实验结果表, 证明 Hybrid+Reranker 比其他方案好。
4. **被追问时**:
   - "为什么用后置匹配而不是让 LLM 生成引用?" → 回答: LLM 会编造引用编号
   - "BM25 索引怎么更新?" → 回答: 从 ChromaDB 全量重建, 秒级
   - "Reflection 会不会误判?" → 回答: 可能, 在局限性章节说明

---

## 10. 设计取舍 + 局限性

| 取舍 | 选择 | 代价 |
|------|------|------|
| 后置匹配 vs LLM 生成引用 | 后置匹配 | 可能漏掉 LLM 用自己的知识做的合理补充 |
| LLM 自检 vs embedding 相似度 | LLM 自检 | 多一次 API 调用, 成本和延迟增加 |
| BM25 内存 vs 磁盘 | 内存 | 重启后需重建, 但秒级可接受 |
| LangChain Agent vs 自定义路由 | LangChain | 框架黑盒, 面试需能讲清楚 AgentExecutor 原理 |

局限性:
- 表格提取依赖 LLM 能力, 复杂表格可能不准。
- Reranker 模型首次下载需要网络。
- 引用相似度阈值 0.6 是经验值, 不同领域可能需要调整。
