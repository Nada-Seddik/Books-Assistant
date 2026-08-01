import io
import re
from pathlib import Path
from typing import List, Dict, Tuple

from pypdf import PdfReader


def normalize_book_name(name: str) -> str:
    normalized = re.sub(r"[\s\-]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", normalized)


def extract_text(filename: str, file_bytes: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_bytes.decode("utf-8", errors="ignore")


def load_documents(book_name: str, uploaded_files: List[Tuple[str, bytes]]) -> List[Dict]:
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
