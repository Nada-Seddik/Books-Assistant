"""
02_preprocessing.py

Text cleanup applied to raw extracted text before chunking: strips
repeated running headers/footers, standalone page numbers, and rejoins
words hyphenated across a line break.

Unchanged from the original project — this logic operates purely on raw
text and has no dependency on retrieval backend, so it needed no changes
when the retrieval stack was replaced with Chroma.
"""

import re
from collections import Counter


def _remove_repeated_lines(lines: list[str], min_repeats: int = 3) -> list[str]:
    short_lines = [line.strip() for line in lines if 0 < len(line.strip()) <= 60]
    counts = Counter(short_lines)
    repeated = {line for line, count in counts.items() if count >= min_repeats}
    return [line for line in lines if line.strip() not in repeated]


def _remove_standalone_page_numbers(lines: list[str]) -> list[str]:
    return [line for line in lines if not re.fullmatch(r"\s*\d{1,4}\s*", line)]


def _dehyphenate(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(raw_text: str) -> str:
    """Run the full cleanup pipeline on one document's raw extracted text."""
    text = _dehyphenate(raw_text)
    lines = text.split("\n")
    lines = _remove_repeated_lines(lines)
    lines = _remove_standalone_page_numbers(lines)
    text = "\n".join(lines)
    return _normalize_whitespace(text)


def clean_documents(documents: list[dict]) -> list[dict]:
    """Apply clean_text to every document's "text" field."""
    for doc in documents:
        doc["text"] = clean_text(doc["text"])
    return documents
