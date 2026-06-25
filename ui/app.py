"""SmartRAG-Agent Streamlit UI

三栏布局:
  - 左侧: 文档管理 (上传/列表/删除)
  - 中间: 问答区 (聊天+引用)
  - 右侧: Agent Trace + Source Viewer
"""

import os
import sys
import tempfile
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ---- 页面配置 ----
st.set_page_config(
    page_title="SmartRAG-Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .citation-found {
        background-color: #e6f3e6;
        border-left: 3px solid #2e7d32;
        padding: 4px 8px;
        margin: 2px 0;
        font-size: 0.85em;
        border-radius: 4px;
    }
    .citation-missing {
        background-color: #fff3e0;
        border-left: 3px solid #e65100;
        padding: 4px 8px;
        margin: 2px 0;
        font-size: 0.85em;
        border-radius: 4px;
    }
    .trace-thought {
        background-color: #e3f2fd;
        padding: 8px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.85em;
    }
    .trace-action {
        background-color: #f3e5f5;
        padding: 8px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.85em;
    }
    .trace-observation {
        background-color: #e8f5e9;
        padding: 8px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# ---- 标题 ----
st.title("📚 SmartRAG-Agent")
st.caption("Agentic RAG 智能文档问答系统")

# ---- 初始化 Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "processing" not in st.session_state:
    st.session_state.processing = False


# ---- 延迟初始化后端组件 ----
@st.cache_resource
def get_backend():
    """初始化后端组件 (Chromadb + Retriever)"""
    from src.vector_store.chroma_store import ChromaStore
    from src.retriever.bm25_retriever import BM25Retriever
    from src.retriever.dense_retriever import DenseRetriever
    from src.retriever.hybrid_retriever import HybridRetriever

    chroma_store = ChromaStore()
    bm25 = BM25Retriever()
    dense = DenseRetriever(chroma_store)

    chunks = chroma_store.get_all_chunks()
    if chunks:
        bm25.build_index(chunks)

    hybrid = HybridRetriever(bm25, dense)

    return {
        "chroma_store": chroma_store,
        "bm25": bm25,
        "hybrid": hybrid,
    }


# 加载后端
try:
    backend = get_backend()
except Exception as e:
    st.error(f"后端初始化失败: {e}")
    st.info("请确保已配置 .env 文件中的 LLM_API_KEY")
    st.stop()


def process_uploaded_file(uploaded_file) -> bool:
    """处理上传的文件: 加载 → 分块 → 存入 ChromaDB → 重建 BM25"""
    from src.loader.pdf_loader import PDFLoader
    from src.loader.docx_loader import DocxLoader
    from src.loader.text_loader import TextLoader
    from src.splitter.semantic_chunker import SemanticChunker

    # 保存上传文件到临时目录
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(uploaded_file.name).suffix,
    ) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        # 选择加载器
        suffix = Path(uploaded_file.name).suffix.lower()
        if PDFLoader.supports(tmp_path):
            loader = PDFLoader()
        elif DocxLoader.supports(tmp_path):
            loader = DocxLoader()
        elif TextLoader.supports(tmp_path):
            loader = TextLoader()
        else:
            st.error(f"不支持的文件类型: {suffix}")
            return False

        # 加载文档
        document = loader.load(tmp_path)

        # 分块
        chunker = SemanticChunker()
        chunks = chunker.split(document)

        if not chunks:
            st.warning("文档内容为空")
            return False

        # 存入 ChromaDB
        backend["chroma_store"].add_chunks(chunks)

        # 重建 BM25 索引
        all_chunks = backend["chroma_store"].get_all_chunks()
        backend["bm25"].build_index(all_chunks)

        # 更新文档列表
        doc_list = backend["chroma_store"].list_documents()
        st.session_state.documents = doc_list

        st.success(f"✅ {uploaded_file.name} — {len(chunks)} 个 chunk 已入库")
        return True

    except Exception as e:
        st.error(f"处理文件失败: {e}")
        return False
    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def delete_document(doc_id: str):
    """删除文档"""
    n = backend["chroma_store"].delete_by_doc_id(doc_id)
    # 重建 BM25 索引
    all_chunks = backend["chroma_store"].get_all_chunks()
    backend["bm25"].build_index(all_chunks)
    # 更新列表
    st.session_state.documents = backend["chroma_store"].list_documents()
    st.success(f"已删除 {n} 个 chunk")


def run_query(question: str) -> dict:
    """执行查询 — 直接使用 RAG Pipeline (1 次 LLM 调用)"""
    from src.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(retriever=backend["hybrid"])
    rag_result = pipeline.query(question, run_reflection=False)

    return {
        "output": rag_result["answer"],
        "citations": rag_result.get("citations", []),
        "retrieved_chunks": rag_result.get("retrieved_chunks", []),
        "reflection": rag_result.get("reflection"),
    }


# ---- 三栏布局 ----
left_col, center_col, right_col = st.columns([1, 2, 1])

# ============================================================
# 左侧栏: 文档管理
# ============================================================
with left_col:
    st.subheader("📁 文档管理")

    # 刷新文档列表
    if st.button("🔄 刷新列表"):
        st.session_state.documents = backend["chroma_store"].list_documents()
        st.rerun()

    # 上传文件
    uploaded_files = st.file_uploader(
        "上传文档",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="支持 PDF / Word / TXT / Markdown 格式",
    )

    if uploaded_files:
        for f in uploaded_files:
            # 检查是否已存在 (按文件名)
            existing_names = {d["filename"] for d in st.session_state.documents}
            if f.name not in existing_names:
                process_uploaded_file(f)
            else:
                st.warning(f"文档已存在: {f.name}")

    st.divider()

    # 已上传文档列表
    st.markdown(f"**已上传文档 ({len(st.session_state.documents)})**")

    if not st.session_state.documents:
        st.info("暂无文档，请上传")

    for doc in st.session_state.documents:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(f"📄 {doc['filename']}")
        with col2:
            if st.button("🗑️", key=f"del_{doc['doc_id']}", help="删除此文档"):
                delete_document(doc["doc_id"])
                st.rerun()

    # 系统状态
    st.divider()
    st.markdown("**系统状态**")
    chunk_total = backend["chroma_store"].chunk_count()
    st.metric("Chunk 总数", chunk_total)

# ============================================================
# 中间栏: 问答区
# ============================================================
with center_col:
    st.subheader("💬 智能问答")

    # 聊天历史
    chat_container = st.container(height=450)

    with chat_container:
        if not st.session_state.messages:
            st.info("👋 欢迎使用 SmartRAG-Agent! 请上传文档后开始提问。")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # 显示引用
                if msg.get("citations"):
                    with st.expander("📎 查看引用来源"):
                        for ci in msg["citations"]:
                            if ci.source_text:
                                source = f"**{ci.filename}**"
                                if ci.page_number:
                                    source += f" — 第{ci.page_number}页"
                                source += f" _(相似度: {ci.similarity})_"

                                st.markdown(
                                    f'<div class="citation-found">{source}<br>'
                                    f'<small>声明: {ci.sentence[:100]}...</small><br>'
                                    f'<small>原文: {ci.source_text[:150]}...</small></div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f'<div class="citation-missing">'
                                    f'⚠️ 未找到直接来源 — "{ci.sentence[:100]}..."'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

    # 输入框
    if prompt := st.chat_input(
        "请输入您的问题...",
        disabled=st.session_state.processing or len(st.session_state.documents) == 0,
    ):
        if not st.session_state.documents:
            st.warning("请先上传文档")
            st.stop()

        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.processing = True
        st.rerun()

# ============================================================
# 右侧栏: 检索详情 + Source Viewer
# ============================================================
with right_col:
    st.subheader("🔍 检索详情")

    last_chunks = None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant":
            if msg.get("retrieved_chunks"):
                last_chunks = msg["retrieved_chunks"]
            break

    if last_chunks:
        st.markdown(f"**检索到 {len(last_chunks)} 个相关片段**")
        for i, chunk in enumerate(last_chunks, start=1):
            score_pct = f"{chunk.score:.1%}" if chunk.score <= 1 else f"{chunk.score:.3f}"
            st.caption(
                f"[{i}] **{chunk.filename}**"
                + (f" p.{chunk.page_number}" if chunk.page_number else "")
                + f" | score: {score_pct}"
            )
    else:
        st.info("等待查询以显示检索结果...")


# ============================================================
# 处理助手消息 (在 rerun 后执行)
# ============================================================
if st.session_state.processing and st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if last_msg["role"] == "user":
        with st.spinner("🔍 正在检索..."):
            try:
                result = run_query(last_msg["content"])
            except Exception as e:
                result = {
                    "output": f"处理出错: {str(e)}",
                    "trace": None,
                    "citations": [],
                    "retrieved_chunks": [],
                }

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["output"],
            "trace": result.get("trace"),
            "citations": result.get("citations", []),
            "retrieved_chunks": result.get("retrieved_chunks", []),
        })
        st.session_state.processing = False
        st.rerun()
