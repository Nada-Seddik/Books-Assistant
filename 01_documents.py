"""
01_documents.py

Document loading: turns uploaded files (PDF or TXT, as raw bytes) into the
document schema used by the rest of the pipeline:

    document_id, title, book, text

Uses in-memory bytes rather than a local folder path, since the deployed
Streamlit app has no access to a user's local filesystem — files come in
through st.file_uploader instead.

normalize_book_name() is preserved from the original project so multiple
books can be indexed and told apart regardless of how a user types the
book's name ("The Art of War" vs "the_art_of_war").
"""

import io
import re
from pathlib import Path
from typing import List, Dict, Tuple

from pypdf import PdfReader


def normalize_book_name(name: str) -> str:
    """Canonical identifier for a book: lowercase, spaces/hyphens -> underscores."""
    normalized = re.sub(r"[\s\-]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", normalized)


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extract raw text from a single PDF or TXT file's bytes."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_bytes.decode("utf-8", errors="ignore")


def load_documents(book_name: str, uploaded_files: List[Tuple[str, bytes]]) -> List[Dict]:
    """Build document rows for one book from its uploaded files.

    uploaded_files: list of (filename, file_bytes) tuples. Each file
    becomes one document row; most books will have one file, but a book
    split across several files (e.g. one PDF per chapter) works the same way.
    """
    if not uploaded_files:
        raise ValueError(f"No files provided for book '{book_name}'.")

    book_key = normalize_book_name(book_name)
    documents = []
    for i, (filename, file_bytes) in enumerate(uploaded_files):
        raw_text = extract_text(filename, file_bytes)
        documents.append({
            "document_id": f"{book_key}_{i}",
            "title": Path(filename).stem,
            "book": book_key,
            "text": raw_text,
        })
    return documents
