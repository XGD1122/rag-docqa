# SmartRAG-Agent — Agentic RAG 智能文档问答系统

一个带 Web UI 的本地文档智能问答系统。支持 PDF/Word/TXT/Markdown 文档上传，基于混合检索 + LLM 生成 + 后置引用匹配，实现可溯源的高质量文档问答。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 LLM 和 Embedding 的 API Key

# 3. 启动
streamlit run ui/app.py
```

浏览器打开 http://localhost:8501，上传文档即可提问。

## API 配置

支持 LLM 和 Embedding 使用不同的 API 提供商，`.env` 配置示例：

```env
# 方案1: 全用一家 (如 OpenAI)
LLM_API_KEY=sk-xxx
LLM_BASE_URL=
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
# EMBEDDING_API_KEY / EMBEDDING_BASE_URL 留空则复用 LLM 配置

# 方案2: LLM + Embedding 分离 (如 DeepSeek + 硅基流动)
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

任何兼容 OpenAI 接口的 API 均可使用。

## 技术架构

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI                        │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ 文档管理  │  │   问答+引用   │  │   检索详情     │  │
│  └──────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                  RAG Pipeline                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ 混合检索  │→│ LLM 生成 │→│ 后置引用匹配      │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 检索链路：四层混合检索

```
Query
  ├─→ BM25 (关键词匹配, rank_bm25)      → top_k × 4
  └─→ Dense (语义匹配, ChromaDB)         → top_k × 4
                │
                ▼
           RRF 融合 (k=60)
           score = Σ 1/(60 + rank_i)
                │
                ▼
           Top-K 最终结果
```

- **BM25**: 纯 Python `rank_bm25` 实现，内存索引，从 ChromaDB 全量重建（秒级）
- **Dense**: ChromaDB + `text-embedding-3-small` (或其他兼容 Embedding)
- **RRF**: Reciprocal Rank Fusion，解决 BM25 和 Dense 分数不可比的问题
- **Reranker**: 可选，BGE-Reranker-v2-m3 cross-encoder 精排（需网络下载模型）

### 引用匹配：后置 Embedding 匹配

```
LLM 生成回答（不含引用标记）
  → 按句拆分 → 逐句 embedding → 与检索到的 chunk 做余弦相似度
  → 相似度 ≥ 0.6: 标注来源 (文档名 + 页码 + 原文)
  → 相似度 < 0.6: 标记 "未找到直接来源"
```

相比让 LLM 直接生成 `[1]` `[2]` 标记，后置匹配基于 embedding 计算，引用来源可验证，避免 LLM 编造引用。

### LLM 自检反思 (Reflection)

```
回答 → LLM 提取关键事实声明
  → 逐声明与 evidence 比对 (LLM 判断是否被支持)
  → 不被支持的声明 → 自动生成补充检索 query
  → 状态: pass / retry / ask_clarification
```

## 项目结构

```
smartrag-agent/
├── src/
│   ├── config.py                  # pydantic-settings 配置 + 数据模型
│   ├── loader/                    # 文档加载器 (PDF/DOCX/TXT/MD)
│   │   ├── base.py                # DocumentLoader 抽象基类
│   │   ├── pdf_loader.py          # PyMuPDF 解析, 保留页码
│   │   ├── docx_loader.py         # python-docx 解析
│   │   └── text_loader.py         # TXT/Markdown 加载
│   ├── splitter/
│   │   └── semantic_chunker.py   # 按标题+段落切分, tiktoken 计数
│   ├── vector_store/
│   │   └── chroma_store.py       # ChromaDB CRUD
│   ├── retriever/
│   │   ├── bm25_retriever.py     # rank_bm25 关键词检索
│   │   ├── dense_retriever.py    # ChromaDB 语义检索
│   │   ├── hybrid_retriever.py   # BM25 + Dense + RRF 融合
│   │   └── reranker.py           # BGE cross-encoder 精排
│   ├── agent/
│   │   ├── tools.py              # 5 个 LangChain Tool
│   │   ├── executor.py           # Agent 封装
│   │   ├── trace.py              # 执行轨迹采集
│   │   └── reflection.py         # LLM 自检
│   ├── rag/
│   │   ├── prompt.py             # Prompt 模板
│   │   ├── citation.py           # 后置引用匹配
│   │   └── pipeline.py           # 完整 RAG 流水线
│   └── evaluation/
│       ├── dataset.py            # 评估集加载
│       ├── retrieval_metrics.py  # Recall@K, MRR, NDCG
│       ├── answer_metrics.py     # Citation Accuracy, Faithfulness
│       └── report.py             # 消融实验报告
├── ui/
│   └── app.py                     # Streamlit 三栏布局
├── examples/
│   ├── sample_docs/               # 测试文档
│   └── qa_eval_set.jsonl          # 30 条标注评估集
├── tests/                         # 单元测试
├── requirements.txt
└── .env.example
```

## 运行评估

```bash
python -m src.evaluation.report
```

对比 4 种检索策略 (BM25-Only / Dense-Only / Hybrid / Hybrid+Reranker)，输出 Recall@K、MRR、NDCG 对比表。

## 技术栈

| 层 | 选型 |
|----|------|
| LLM | OpenAI / DeepSeek / 任意 OpenAI 兼容接口 |
| Embedding | OpenAI / 硅基流动 / 任意 OpenAI 兼容接口 |
| 向量库 | ChromaDB |
| Agent 框架 | LangChain `create_agent` |
| BM25 | rank_bm25 |
| Reranker | BGE-Reranker-v2-m3 (FlagEmbedding) |
| 文档解析 | PyMuPDF + python-docx |
| UI | Streamlit |
| 配置 | pydantic-settings |


