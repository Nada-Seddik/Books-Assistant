"""
03_chunking.py

Word-based sliding-window chunking, unchanged from the original project.
chunk_id is now a string (e.g. "the_art_of_war_0_3") instead of a plain
integer, since Chroma requires string IDs and a book-scoped, globally
unique ID lets multiple books share one physical vector store safely.
"""


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(documents: list[dict], chunk_size: int = 120, overlap: int = 30) -> list[dict]:
    """Chunk every document into rows ready for embedding and storage."""
    rows = []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)):
            rows.append({
                "chunk_id": f"{doc['document_id']}_{i}",
                "document_id": doc["document_id"],
                "title": doc["title"],
                "book": doc["book"],
                "text": chunk,
                "search_text": f"{doc['title']}: {chunk}",
            })
    return rows
