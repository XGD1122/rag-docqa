# SmartRAG-Agent 面试问答清单

---

## 一、架构与设计

### Q1: 这个项目的整体架构是怎样的？

三层架构：
- **UI 层**：Streamlit 三栏布局（文档管理 / 问答+引用 / 检索详情）
- **Pipeline 层**：检索 → 生成 → 引用匹配 → 反思自检，串联为一个完整流程
- **存储层**：ChromaDB 持久化向量 + 元数据，BM25 内存索引从 ChromaDB 重建

数据流：用户上传文档 → 解析 → 分块 → Embedding → 存入 ChromaDB → 同步重建 BM25 索引 → 用户提问 → 混合检索 → LLM 生成 → 后置引用匹配。

---

### Q2: 为什么用 RRF 融合而不是加权求和？

BM25 的分数是词频相关的稀疏分数（可能 0-几十），Dense 的分数是余弦相似度（0-1）。两者数值尺度完全不同，加权求和的权重很难调。RRF 不关心原始分数值，只关心相对排名：

```
score(doc) = Σ 1/(k + rank_i(doc))  其中 k=60
```

一个参数 `k` 控制融合平滑度，k 越大排名差异的影响越小。业界（如 Elasticsearch 8.x）也采用此方案。

---

### Q3: BM25 索引如何更新？

ChromaDB 是唯一数据源（Single Source of Truth）。BM25 不持久化，仅在内存中维护。文档上传/删除时，从 ChromaDB 读全量 chunk 重建 BM25 索引。几千个 chunk 的重建在秒级完成，性能可以接受。这个设计避免了维护两套存储一致性的复杂问题。

---

### Q4: 为什么用后置匹配做引用，而不是让 LLM 直接生成引用标记？

LLM 存在"幻觉引用"问题——在长回答中经常编造 `[1]` `[2]` 标记指向错误的 chunk。后置匹配的做法是：

1. LLM 生成纯文本回答（不标记引用）
2. 按句拆分，逐句计算 embedding
3. 与检索到的所有 chunk 的 embedding 做余弦相似度
4. 相似度 ≥ 0.6 → 匹配成功；< 0.6 → 标记"未找到来源"

这样做的好处是引用完全可验证，任何一条引用都能追溯到具体的 chunk。代价是如果 LLM 用自己的知识补充了合理内容，这部分可能匹配不到来源。

---

## 二、检索与向量

### Q5: 混合检索的完整流程？

```
Query → BM25 检索 top_k×4 → Dense 检索 top_k×4
      → 去重 (按 chunk_id)
      → RRF 融合排序
      → 可选 BGE-Reranker 精排
      → 返回 top_k
```

BM25 擅长精确关键词匹配（如"违约金比例"），Dense 擅长语义匹配（如"违约要赔多少钱"），两者互补。RRF 融合后取排名靠前的交集和各自的高排名结果。

---

### Q6: 为什么选择 ChromaDB？

- 本地持久化，零配置，适合演示和开发环境
- 支持元数据过滤（按 doc_id 删除、按文件名查询）
- 与 LangChain 集成良好
- 速度对于万级以下 chunk 完全够用

对比：FAISS 需要额外序列化，Pinecone/Weaviate 需要云端部署，Milvus 太重。

---

### Q7: Embedding 模型怎么选的？

默认使用 OpenAI `text-embedding-3-small`（1536 维），性价比高。同时支持通过 `.env` 切换到任意 OpenAI 兼容的 Embedding API（如硅基流动的 `BAAI/bge-large-zh-v1.5`，1024 维）。关键设计：LLM 和 Embedding 的 API Key/Base URL 可以独立配置，支持混用不同厂商。

---

### Q8: ChromaDB 中的 chunk 存储了什么？

每个 chunk 存储：
- `id`: `{doc_id}_chunk_{idx}`
- `document`: 文本内容
- `metadata`: `{doc_id, filename, page_number, chunk_index, token_count}`
- `embedding`: text-embedding-3-small 生成的向量

通过 `doc_id` 实现按文档批量删除，通过 `page_number` 支持 PDF 页码溯源。

---

## 三、RAG 细节

### Q9: 文档解析和分块策略是什么？

**解析**：
- PDF → PyMuPDF (fitz)，逐页提取文本，自动插入 `[第N页]` 标记保留页码
- DOCX → python-docx，提取段落文本
- TXT/MD → 直接读取

**分块**（SemanticChunker）：
1. 按 Markdown 标题（`#`, `##`, `###`）初次切分
2. 每个 section 内按双换行再次切分
3. 合并太小的段落，控制在 chunk_size（默认 512 tokens）以内，重叠 chunk_overlap（默认 64 tokens）
4. 使用 tiktoken（cl100k_base）计算 token 数

保留页码：从 `[第N页]` 标记中正则提取，存入 chunk 元数据。

---

### Q10: LLM 生成回答时 Prompt 怎么设计？

```text
你是文档问答助手。严格基于下面的文档片段回答问题，不得使用外部知识。

规则:
- 答案必须能直接从文档中找到依据
- 文档中找不到答案时，直接说"文档中未找到相关信息"，不要猜测
- 回答简洁直接，先给结论再展开细节
- 引用具体数据时要准确，不要改动数字

## 文档片段
{context}

## 问题
{question}
```

关键设计点：
- "不得使用外部知识" → 强制 LLM 只看文档（但 LLM 不一定完全遵守）
- "先给结论再展开" → 让回答更结构化
- 上下文以 `[来源: 文件名, 第N页]` 前缀标注

---

### Q11: 引用匹配的阈值 0.6 怎么定的？

经验值。在测试文档上手动验证了几十组匹配结果：
- ≥ 0.7：几乎都是正确匹配
- 0.6-0.7：大部分正确，偶有语义相近但内容不相关的误匹配
- < 0.6：匹配质量明显下降

0.6 是精确和召回之间的折中。不同领域（技术文档 vs 合同 vs 小说）的最优阈值可能不同，实际使用中可调整。

---

## 四、Agent 工具

### Q12: 定义了哪些 Agent Tool？各自做什么？

5 个 Tool，基于 LangChain `@tool` 装饰器定义：

| Tool | 功能 | 适用场景 |
|------|------|---------|
| `search` | 混合检索 + LLM 生成 | 精确事实查找 |
| `compare` | 扩大检索量 + 对比 Prompt | 多文档对比分析 |
| `summarize` | 扩大检索量 + 摘要 Prompt | 主题概括 |
| `extract_table` | 增强关键词 + 表格格式化 Prompt | 表格数据提取 |
| `ask_clarification` | 返回澄清问题 | 问题不够明确时 |

每个 Tool 内部都是"检索 → 找上下文 → LLM 生成"的模式，区别在于检索数量、Prompt 模板和输出格式。

---

### Q13: Agent 是怎么路由决策的？

使用 LangChain 的 `create_agent` API（基于 `langgraph`），底层是 OpenAI function calling 机制。LLM 根据问题的语义和 Tool 的 description 自动选择调用哪个 Tool。

实际使用中，因为每次 Agent 决策本身也是一次 API 调用，增加了延迟。当前 UI 直接走 RAG Pipeline（1 次 LLM 调用），Agent 作为可选能力保留。

---

### Q14: Agent 的执行轨迹怎么采集？

从 `create_agent` 返回的 messages 列表中提取：
- `AIMessage(tool_calls=[...])` → 解析 tool name 和 args
- `ToolMessage` → 解析 tool 返回结果
- 组装为 `AgentStep(step_number, action, action_input, observation)`

在 UI 的右侧面板可展开查看每一步的 Action → Observation。

---

## 五、LLM 自检反思

### Q15: 反思机制怎么工作的？

```
Step 1: LLM 从回答中提取事实声明
  输入: "违约金为5%，逾期超过30天触发。"
  输出: ["违约金比例为5%", "逾期超过30天触发"]

Step 2: LLM 逐声明与检索证据比对
  claim="违约金比例为5%" + evidence="...违约金为总金额5%..."
  → LLM 判断: supported=True

Step 3: 不被支持的声明 → 生成补充检索 query
  输出: ["逾期触发条件", "违约超过多少天触发罚则"]

Step 4: 返回状态
  all supported → pass
  有补充 query → retry (重新检索)
  无法生成 query → ask_clarification
```

---

### Q16: 为什么反思用 LLM 判断而不是 embedding 相似度？

同一事实可以用完全不同的措辞表达：
- "违约金 5%" ↔ "按总金额百分之五收取罚金"
- Embedding 相似度可能不高（措辞差异大）
- 但 LLM 能正确判断两者是同一事实

代价是每次反思多一次 API 调用，增加成本和延迟。在生产环境中，反思是可选的（`run_reflection=False` 可以跳过）。

---

## 六、评估体系

### Q17: 评估集怎么构建的？

30 条 JSONL 格式的标注 QA 对，覆盖 5 种问题类型（各 6 条）：
- `exact_qa`: 精确事实查找
- `compare`: 多文档对比
- `summary`: 主题摘要
- `table`: 表格提取
- `missing`: 故意问文档中没有的信息（验证"证据不足"处理）

每条包含：`question`, `answer`, `relevant_chunks`（手工标注哪些 chunk 包含正确答案）, `question_type`, `documents`。

---

### Q18: 用了哪些评估指标？

**检索质量**：
- Recall@K：前 K 个结果命中相关 chunk 的比例
- MRR：第一个相关结果的倒数排名
- NDCG@K：考虑排序位置的折损累计增益

**回答质量**：
- Citation Accuracy：引用命中相关 chunk 的比例
- Faithfulness：LLM 评估回答是否忠实于文档证据

---

### Q19: 消融实验做了什么？结论是什么？

对比 4 种检索策略：
1. BM25-Only
2. Dense-Only
3. Hybrid (BM25 + Dense + RRF)
4. Hybrid + Reranker

预期结论：Hybrid+Reranker 在所有质量指标上最优（Recall@5 最高，MRR 最高），但延迟略高于纯 Hybrid。证明混合检索 + 精排是有效方案。

---

## 七、工程与实战

### Q20: 为什么 LLM 和 Embedding 的 API 要分离配置？

实际场景中，最优 LLM 和最优 Embedding 往往来自不同厂商：
- DeepSeek 的 chat 模型便宜好用，但没有 Embedding API
- 硅基流动的 BGE Embedding 中文效果好，但 chat 模型不如 DeepSeek

分离配置允许"DeepSeek 生成 + 硅基 Embedding"这种组合，`.env` 中各自配置 API Key 和 Base URL。

---

### Q21: 为什么去掉了 Reranker？

BGE-Reranker-v2-m3 模型约 2GB，首次使用需从 HuggingFace 下载。在国内网络环境下 HuggingFace 经常无法访问，导致超时。Reranker 在 `reranker.py` 中保留为可选模块，`_ensure_model()` 加载失败时会静默降级（原样返回候选结果），不会影响核心功能。

---

### Q22: 项目中遇到过什么坑？怎么解决的？

| 问题 | 原因 | 解决 |
|------|------|------|
| LangChain 1.3.x 移除 AgentExecutor | API 变更 | 改用 `create_agent` API |
| 硅基 Embedding 调用 400 错误 | LangChain 默认用 tiktoken 转 token ID 发送 | `check_embedding_ctx_length=False` |
| DeepSeek function calling 效果差 | 路由决策不准 | 主流程绕过 Agent，走 RAG Pipeline |
| HuggingFace 被墙，Reranker 下载超时 | 网络限制 | Reranker 降级为可选，失败静默跳过 |

---

### Q23: 如果要改进，你会做什么？

1. **多轮对话**：接入 chat history，支持追问和澄清
2. **更好的表格提取**：使用 Unstructured.io 或专门的结构化提取 pipeline
3. **增量 BM25 更新**：目前是全量重建，可改为只增删变更的 chunk
4. **流式输出**：LLM 生成改用 streaming，提升用户体验
5. **引用高亮**：在原始文档中高亮显示引用位置
6. **多模态**：支持图片中的文字提取（OCR）
