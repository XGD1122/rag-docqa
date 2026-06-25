"""ChromaDB 向量存储

ChromaDB 是整个系统的唯一数据源 (Single Source of Truth)。
BM25 索引从 ChromaDB 中的数据重建。
"""

from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from src.config import Chunk, SearchResult, settings


class ChromaStore:
    """ChromaDB 向量存储封装"""

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_name = "documents"
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 构建 embedding 函数
        embed_kwargs = {"model": settings.embedding_model}
        if settings.effective_embedding_base_url:
            embed_kwargs["base_url"] = settings.effective_embedding_base_url
        self._embeddings = OpenAIEmbeddings(
            openai_api_key=settings.effective_embedding_api_key,
            check_embedding_ctx_length=False,
            **embed_kwargs,
        )

    # ---- CRUD ----

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """批量添加 chunk 到 ChromaDB"""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "filename": c.filename,
                "page_number": c.page_number or 0,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
            }
            for c in chunks
        ]

        # 使用 LangChain embedding 生成向量
        embeddings = self._embeddings.embed_documents(documents)

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def delete_by_doc_id(self, doc_id: str) -> int:
        """删除指定文档的所有 chunk，返回删除数量"""
        existing = self._collection.get(
            where={"doc_id": doc_id},
        )
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def search(
        self, query: str, top_k: int | None = None
    ) -> List[SearchResult]:
        """向量语义检索"""
        k = top_k or settings.top_k
        query_embedding = self._embeddings.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        search_results = []
        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0
            # ChromaDB 使用余弦距离, 转换为相似度 (0-1)
            similarity = 1.0 - distance if distance else 1.0

            search_results.append(SearchResult(
                chunk_id=chunk_id,
                content=results["documents"][0][i] if results["documents"] else "",
                score=similarity,
                filename=metadata.get("filename", "unknown"),
                page_number=metadata.get("page_number") or None,
                source_scores={"dense": similarity},
            ))

        return search_results

    def get_all_chunks(self) -> List[Chunk]:
        """获取所有 chunk (用于重建 BM25 索引)"""
        results = self._collection.get(
            include=["documents", "metadatas"],
        )

        if not results["ids"]:
            return []

        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i] if results["metadatas"] else {}
            content = results["documents"][i] if results["documents"] else ""

            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=metadata.get("doc_id", ""),
                filename=metadata.get("filename", ""),
                content=content,
                page_number=metadata.get("page_number") or None,
                chunk_index=metadata.get("chunk_index", i),
                token_count=metadata.get("token_count", 0),
            ))

        return chunks

    def list_documents(self) -> List[dict]:
        """列出已存储的文档列表 (去重)"""
        results = self._collection.get(include=["metadatas"])
        if not results["metadatas"]:
            return []

        seen = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id", "")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                }
        return list(seen.values())

    def chunk_count(self) -> int:
        """返回当前 chunk 总数"""
        return self._collection.count()
