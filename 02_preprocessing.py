import re
from collections import Counter


def _strip_glyph_artifacts(text: str) -> str:
    return re.sub(r"(?:/G[0-9A-Fa-f]{2}){3,}", " ", text)


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
    text = _strip_glyph_artifacts(raw_text)
    text = _dehyphenate(text)
    lines = text.split("\n")
    lines = _remove_repeated_lines(lines)
    lines = _remove_standalone_page_numbers(lines)
    text = "\n".join(lines)
    return _normalize_whitespace(text)


def clean_documents(documents: list[dict]) -> list[dict]:
    for doc in documents:
        doc["text"] = clean_text(doc["text"])
    return documents
