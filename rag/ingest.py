"""
Multi-format document loader.
Supports: PDF, TXT, Markdown, web URLs.
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader


def load_documents(source: str) -> list[Document]:
    """
    Load documents from a PDF path, plain text file, or URL.

    Args:
        source: File path (PDF / TXT / MD) or an http(s) URL.

    Returns:
        List of LangChain Document objects.
    """
    source = source.strip()

    if source.startswith(("http://", "https://")):
        loader = WebBaseLoader(source)
        return loader.load()

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()

    if ext in (".txt", ".md"):
        return TextLoader(str(path), encoding="utf-8").load()

    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: .pdf, .txt, .md, or a URL."
    )
