"""PDF 文档加载器

使用 PyMuPDF (fitz) 解析 PDF，按页提取文本并保留页码元数据。
"""

import uuid
from pathlib import Path
from src.loader.base import DocumentLoader
from src.config import Document


class PDFLoader(DocumentLoader):
    """PDF 文档加载器"""

    @staticmethod
    def supports(file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def load(self, file_path: str) -> Document:
        import fitz  # PyMuPDF

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        doc = fitz.open(str(path))
        pages_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages_text.append(f"[第{page_num + 1}页]\n{text}")
            else:
                pages_text.append(f"[第{page_num + 1}页]")

        full_text = "\n\n".join(pages_text)
        doc.close()

        return Document(
            doc_id=str(uuid.uuid4()),
            filename=path.name,
            file_type="pdf",
            raw_text=full_text,
            page_count=len(pages_text),
            metadata={"file_path": str(path.absolute())},
        )
