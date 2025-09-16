# src/ingest.py
# -*- coding: utf-8 -*-
"""
Tiny ingest utility for the RAG pipeline.

- Walks a data directory
- Reads .md/.txt as plain text, .csv as flattened text
- Splits into overlapping character chunks (good enough for MiniLM on CPU)
- Emits a list of dicts: {id, text, source, title, chunk_idx}

Intentionally simple — easy to debug, no heavyweight deps.
"""

from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import List, Dict

# rough char target for ~400–500 tokens (MiniLM): ~1200–1500 chars
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TEXT_EXTS = {".md", ".txt"}
CSV_EXTS = {".csv"}


def _slurp_text(path: Path) -> str:
    """
    Read a text file as UTF-8; fall back to latin-1 if someone saved weird encodings.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with path.open("r", encoding="latin-1", errors="ignore") as f:
            return f.read()


def _csv_to_text(path: Path, max_rows: int = 3000) -> str:
    """
    Flatten CSV into a line-wise text that still looks natural to a reader.
    We don’t need pretty tables for retrieval; key=value pairs are fine.

    Example:
      CSV_HEADERS: a, b, c
      a=1; b=2; c=3
      ...
    """
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        headers = [h.strip() for h in headers]

        # header line first (helps retrieval anchors)
        lines.append(f"CSV_HEADERS: {', '.join(headers)}")

        for i, row in enumerate(reader):
            if i >= max_rows:
                break  # keep things sane
            fields = [str(x).strip() for x in row]
            # pair up header=value until one side runs out
            upto = min(len(headers), len(fields))
            if upto == 0:
                continue
            pairs = [f"{headers[idx]}={fields[idx]}" for idx in range(upto)]
            lines.append("; ".join(pairs))

    return "\n".join(lines)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple char-based chunking with overlap. Good enough for MiniLM on CPU.
    Not trying to be clever; this keeps context reasonably intact.
    """
    text = (text or "").strip()
    n = len(text)
    if n == 0:
        return []
    if n <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        # step forward leaving some overlap behind
        start = max(0, end - overlap)
    return chunks


def _title_from_path(path: Path) -> str:
    """
    Title heuristic: parent folder + file stem, e.g. `contracts / supplier_terms`.
    """
    parts = list(path.parts)
    if len(parts) >= 2:
        return f"{parts[-2]} / {path.stem}"
    return path.stem


def discover_and_chunk(data_dir: str = "data") -> List[Dict]:
    """
    Walk `data_dir` and return a list of chunk dicts:
      {
        "id": f"{posix_path}::chunk{idx}",
        "text": chunk_text,
        "source": posix_path,
        "title": title_str,
        "chunk_idx": idx
      }

    Skips unknown file types. Add PDF later if/when you need it.
    """
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs: List[Dict] = []
    file_count = 0

    # Note: not sorting files; ordering isn't important for retrieval.
    for root, _, files in os.walk(base):
        for fname in files:
            path = Path(root) / fname
            ext = path.suffix.lower()

            # skip hidden dotfiles (macOS likes to sprinkle .DS_Store everywhere)
            if path.name.startswith("."):
                continue

            if ext in TEXT_EXTS:
                raw = _slurp_text(path)
            elif ext in CSV_EXTS:
                raw = _csv_to_text(path)
            else:
                # unknown type — shrug and move on (PDF support can be added here)
                continue

            title = _title_from_path(path)
            chunks = _chunk_text(raw)

            # if a file is empty (or got eaten by encodings), skip quietly
            if not chunks:
                continue

            for idx, ch in enumerate(chunks):
                doc_id = f"{path.as_posix()}::chunk{idx}"
                docs.append(
                    {
                        "id": doc_id,
                        "text": ch,
                        "source": path.as_posix(),
                        "title": title,
                        "chunk_idx": idx,
                    }
                )

            file_count += 1

    print(f"[ingest] Processed files: {file_count}, produced chunks: {len(docs)}")
    return docs


if __name__ == "__main__":
    # quick manual run/debug
    preview = discover_and_chunk("data")
    for d in preview[:3]:
        print("----")
        print(d["id"], d["title"])
        print(d["text"][:240].replace("\n", " "), "...")
