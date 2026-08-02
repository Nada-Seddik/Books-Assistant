from sentence_transformers import SentenceTransformer  # type: ignore[import]

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of chunk texts (used when building the vector store).
    normalize_embeddings=True scales each vector to unit length, so cosine
    distance in Chroma (1 - cosine_similarity) stays in a fixed, predictable
    [0, 2] range regardless of corpus. This is what makes a fixed similarity
    threshold in 06_retrieve_context.py meaningful across different books.
    """
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string (used at retrieval time).
    Uses the same normalize_embeddings=True path as embed_texts, so query
    and chunk vectors are on the same footing for cosine distance.
    """
    return embed_texts([query])[0]
