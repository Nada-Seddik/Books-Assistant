"""
04_vector_representation.py

Turns text into vector embeddings using a local sentence-transformers
model (no external embedding API needed, keeping this step free and
offline-capable). This module is new: the original project used
embeddings only as one input to a hybrid TF-IDF/BM25/semantic blend.
Here, embeddings are the sole retrieval signal, matching the required
pipeline (vector representation -> vector store -> context retrieval).

Kept independent of Streamlit so this module can be tested or reused on
its own; streamlit_app.py is responsible for caching the loaded model
across reruns (via st.cache_resource) for performance.
"""

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Lazily load and cache the embedding model for this process."""
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
