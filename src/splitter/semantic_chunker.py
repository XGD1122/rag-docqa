"""语义分块器

按标题/段落结构切分文档，保留页码元数据。
"""

import re
from typing import List
import tiktoken
from src.config import Document, Chunk, settings


class SemanticChunker:
    """按文档语义结构进行分块"""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None

    def _count_tokens(self, text: str) -> int:
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # 回退: 简单估算 (中文字符 ~1.5 token, 英文单词 ~1.3 token)
        return len(text)

    def _split_into_sections(self, text: str) -> List[str]:
        """按标题标记和双换行将文本拆分为段落"""
        # 按标题分割 (# ## ### 等)
        sections = re.split(r"\n(?=#{1,4}\s)", text)

        result = []
        for section in sections:
            # 每个段落内再按双换行拆分
            paragraphs = re.split(r"\n\s*\n", section)
            result.extend(p for p in paragraphs if p.strip())

        return result

    def _merge_small_sections(self, sections: List[str]) -> List[str]:
        """合并过小的段落并控制 chunk 大小"""
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for section in sections:
            section_tokens = self._count_tokens(section)

            if current_tokens + section_tokens > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # overlap: 保留上一个 section 的内容
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap * 2:]
                    current_chunk = overlap_text + "\n\n" + section
                    current_tokens = self._count_tokens(current_chunk)
                else:
                    current_chunk = section
                    current_tokens = section_tokens
            else:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
                current_tokens += section_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def split(self, document: Document) -> List[Chunk]:
        """将文档切分为 Chunk 列表"""
        sections = self._split_into_sections(document.raw_text)
        merged = self._merge_small_sections(sections)

        chunks = []
        for idx, content in enumerate(merged):
            # 提取页码信息 (来自 PDF 加载器的 [第N页] 标记)
            page_number = self._extract_page_number(content)

            chunk = Chunk(
                chunk_id=f"{document.doc_id}_chunk_{idx}",
                doc_id=document.doc_id,
                filename=document.filename,
                content=content,
                page_number=page_number,
                chunk_index=idx,
                token_count=self._count_tokens(content),
                metadata={"file_type": document.file_type},
            )
            chunks.append(chunk)

        return chunks

    def _extract_page_number(self, text: str) -> int | None:
        """从文本中提取 [第N页] 标记的页码"""
        match = re.search(r"\[第(\d+)页\]", text)
        if match:
            return int(match.group(1))
        return None
