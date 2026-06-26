# SmartRAG-Agent 简历项目描述

## 一句话概述

实现了一个 Agentic RAG 智能文档问答系统，采用 BM25+Dense+RRF 混合检索、后置 Embedding 引用匹配、LLM 自检反思机制，支持多格式文档上传与可溯源问答，并构建了评估集进行消融实验验证。

## 项目描述 (200字版本)

基于 LangChain 和 ChromaDB 构建的 Agentic RAG 系统，支持 PDF/Word/TXT 文档的智能问答。核心创新包括：

1. **四层混合检索**：BM25 关键词检索 + Dense 语义检索 + RRF 融合 + BGE-Reranker 精排，相比单一检索方式 Recall@5 提升显著
2. **后置引用匹配**：LLM 生成回答后，逐句计算 Embedding 与检索片段的余弦相似度来定位来源（阈值 0.6），避免 LLM 编造引用编号的幻觉问题
3. **LLM 自检反思**：提取回答中的事实声明，逐条与证据比对，自动发现未被文档支持的声明并生成补充检索 query
4. **Agent 工具路由**：定义 search/compare/summarize/extract_table/ask_clarification 五个 Tool，由 LangChain Agent 根据问题类型自动选择合适的工具
5. **评估体系**：构建 30 条标注 QA 集（覆盖精确查找/对比/摘要/表格/缺失 5 类场景），实现 Recall@K/MRR/NDCG/Citation Accuracy 指标，完成 4 种检索策略的消融实验

技术栈：Python, LangChain, ChromaDB, OpenAI API, Streamlit, rank_bm25, PyMuPDF。

## 项目亮点 (面试口述)

- **混合检索不是简单拼接**：BM25 和 Dense 分数不在同一量级，采用 RRF（Reciprocal Rank Fusion, k=60）基于排名融合，只需调一个参数
- **引用可验证**：不是让 LLM 生成 `[1]` `[2]` 标记（会编造），而是后置做 embedding 相似度匹配，每个引用指向具体的文档名+页码+原文
- **反思不是噱头**：LLM 判断同事实的不同措辞（如"违约金 5%"和"按总金额百分之五收取罚金"），embedding 相似度做不到，但 LLM 可以
- **BM25 索引更新策略**：ChromaDB 是唯一数据源，BM25 只在内存中，文档变更时从 ChromaDB 全量重建（千级 chunk 秒级完成）
- **有评估数据支撑**：不是"感觉效果好"，构建了 30 条标注 QA 的评估集，跑消融实验证明 Hybrid+Reranker 优于单一检索

## 技术关键词

RAG · 混合检索 · BM25 · RRF · ChromaDB · LangChain · Agent · 引用匹配 · LLM 自检 · 消融实验 · Streamlit · Embedding
