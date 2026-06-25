"""SmartRAG-Agent 配置管理

使用 pydantic-settings 从 .env 文件和环境变量加载配置。
同时定义系统核心数据模型。
"""

from dataclasses import dataclass, field
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置，从 .env 和环境变量加载

    LLM 和 Embedding 可以使用不同的 API:
      - LLM:     LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
      - Embedding: EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL
      - Embedding 留空则自动复用 LLM 的配置
    """

    # LLM 配置
    llm_api_key: str = "sk-xxx"
    llm_base_url: str = ""
    llm_model: str = "gpt-4o-mini"

    # Embedding 配置 (留空复用 LLM)
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"

    chroma_persist_dir: str = "./chroma_data"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    rrf_k: int = 60

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # 兼容旧版 .env 中的 OPENAI_API_KEY 等字段名
        "extra": "ignore",
    }

    @property
    def effective_llm_api_key(self) -> str:
        return self.llm_api_key

    @property
    def effective_llm_base_url(self) -> str:
        return self.llm_base_url

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key

    @property
    def effective_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.llm_base_url


# 全局单例
settings = Settings()


# ============================================================
# 核心数据模型
# ============================================================

@dataclass
class Document:
    """上传文档的内部表示"""
    doc_id: str
    filename: str
    file_type: str          # "pdf" | "docx" | "txt" | "md"
    raw_text: str
    page_count: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """切分后的文本块 —— ChromaDB 中存储的最小单元"""
    chunk_id: str
    doc_id: str
    filename: str
    content: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """统一检索结果"""
    chunk_id: str
    content: str
    score: float
    filename: str
    page_number: Optional[int] = None
    source_scores: dict = field(default_factory=dict)


@dataclass
class Citation:
    """单条引用"""
    sentence: str
    source_chunk_id: Optional[str] = None
    source_text: Optional[str] = None
    similarity: float = 0.0
    filename: Optional[str] = None
    page_number: Optional[int] = None


@dataclass
class ReflectionResult:
    """反思检查结果"""
    status: str             # "pass" | "retry" | "ask_clarification"
    claims: list = field(default_factory=list)
    unsupported_claims: list = field(default_factory=list)
    followup_queries: list = field(default_factory=list)
