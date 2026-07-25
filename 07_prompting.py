"""
07_prompting.py

Builds the prompt sent to the LLM and calls OpenRouter's chat completions
API to generate an answer. Ported from the original project's prompt
builder; the LLM call itself is new, replacing the local Ollama call with
OpenRouter, since a deployed Streamlit app can't reach a local Ollama
server.

Per the project's API key rules: OPENROUTER_API_KEY and OPENROUTER_MODEL
are read from environment variables here, never hardcoded. They're module
attributes (not constants pulled inline) specifically so streamlit_app.py
can overwrite them from st.secrets after import, for deployment, without
this module needing to know anything about Streamlit.
"""

import os

import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT_TEMPLATE = """You are a careful assistant answering questions about "{book_name}".

Rules:
- Answer using only the context provided below.
- If the context does not contain the answer, say so plainly instead of guessing.
- Reference which source(s) you used when relevant.
- Keep the answer concise and directly responsive to the question.

Context:
{context_text}

Question: {query}

Answer:"""

DESCRIPTION_PROMPT_TEMPLATE = """In 2-3 sentences, describe what the book "{book_name}" is about, \
based on this excerpt from its opening:

{sample_text}

Description:"""

DESCRIPTION_SAMPLE_WORDS = 400  # how much of the book's opening text to sample


def build_prompt(context_package: dict, book_name: str) -> str:
    return PROMPT_TEMPLATE.format(
        book_name=book_name,
        context_text=context_package["context_text"] or "(no relevant context found)",
        query=context_package["query"],
    )


def _call_llm(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Provide it via a local .env "
            "(never committed) or, when deployed, via Streamlit secrets."
        )

    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def generate_answer(prompt: str) -> str:
    return _call_llm(prompt)


def generate_description(book_name: str, sample_text: str) -> str:
    """Generate a short blurb for a book from a sample of its opening text.

    Called once per book, right after indexing, and the result is stored
    as Chroma collection metadata (see 05_create_chroma_store.py) rather
    than regenerated on every view.
    """
    sample = " ".join(sample_text.split()[:DESCRIPTION_SAMPLE_WORDS])
    prompt = DESCRIPTION_PROMPT_TEMPLATE.format(book_name=book_name, sample_text=sample)
    return _call_llm(prompt)
