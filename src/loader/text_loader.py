"""纯文本 / Markdown 文档加载器"""

import uuid
from pathlib import Path
from src.loader.base import DocumentLoader
from src.config import Document


class TextLoader(DocumentLoader):
    """TXT / Markdown 文档加载器"""

    SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}

    @staticmethod
    def supports(file_path: str) -> bool:
        return Path(file_path).suffix.lower() in TextLoader.SUPPORTED_SUFFIXES

    def load(self, file_path: str) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()
        raw_text = path.read_text(encoding="utf-8")

        file_type = "md" if suffix in (".md", ".markdown") else "txt"

        return Document(
            doc_id=str(uuid.uuid4()),
            filename=path.name,
            file_type=file_type,
            raw_text=raw_text,
            metadata={"file_path": str(path.absolute())},
        )
