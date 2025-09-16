import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# src/rag_chat.py
    
from typing import List, Dict, Tuple
import time
import re
import logging

from embed_store import load_index_and_meta, search_standalone

# simple logger (won't spam Streamlit)
logger = logging.getLogger("rag_chat")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# module cache
_index = None
_meta = None
_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- tiny helpers ----------
def _keyword_score(text: str, query: str) -> int:
    """
    Crude keyword bump: count overlaps for tokens like SKU codes / numbers.
    Helps disambiguate MiniLM when queries include codes or HSNs.
    """
    q_tokens = re.findall(r"[A-Za-z0-9\-]+", (query or "").lower())
    doc = (text or "").lower()
    hits = 0
    for t in q_tokens:
        if len(t) >= 3 and t in doc:
            hits += 1
    return hits

def _cheap_clean(text: str, max_len: int = 240) -> str:
    t = (text or "").strip().replace("  ", " ")
    return (t[:max_len] + "...") if len(t) > max_len else t

# -----------------------------------

def init_store():
    """Lazy load FAISS & metadata; rebuild is handled in the app sidebar."""
    global _index, _meta
    if _index is None or _meta is None:
        logger.info("Loading index+meta ...")
        t0 = time.time()
        _index, _meta = load_index_and_meta()
        logger.info(f"Loaded. chunks={len(_meta)} in {time.time()-t0:.2f}s")

def retrieve(query: str, top_k: int = 4) -> List[Dict]:
    """Semantic search + tiny keyword re-rank."""
    if not query or len(query.strip()) < 2:
        return []
    init_store()

    try:
        # Use the standalone search function with the correct signature
        hits = search_standalone(_index, _meta, query, top_k=top_k, model_name=_model_name)

        # re-score with a small keyword bump
        rescored = []
        for h in hits:
            bump = 0.02 * _keyword_score(h.get("text_preview", ""), query)
            rescored.append((h["score"] + bump, h))

        # tiny bias towards policies/contracts (less fluff)
        boosted = []
        for sc, h in rescored:
            src = h.get("source", "")
            if "/policies/" in src or "/contracts/" in src:
                sc += 0.05
            boosted.append((sc, h))

        boosted.sort(key=lambda x: x[0], reverse=True)
        hits = [h for _, h in boosted]
        return hits

    except Exception as e:
        logger.error(f"Error during document retrieval: {e}")
        return []

def build_answer(query: str, hits: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Short extractive answer:
    - takes top hit (already re-ranked)
    - cleans it to 1–2 lines
    - adds 1–2 supporting bullets (also trimmed)
    LLM (generate.py) will polish further if enabled.
    """
    if not hits:
        return ("Sorry, I couldn't find a clear answer in the indexed documents. "
                "Try a simpler phrasing or rebuild the index after adding files.", [])

    # main line
    main = _cheap_clean(hits[0].get("text_preview", ""), max_len=220)

    # supporting lines (at most 2)
    support = []
    for h in hits[1:3]:
        support.append(_cheap_clean(h.get("text_preview", ""), max_len=160))

    # stitch like a human would (not too verbose)
    parts = [main]
    if support:
        parts.append("")
        parts.append("Supporting:")
        for s in support:
            parts.append(f"- {s}")

    answer = "\n".join(parts).strip()
    return answer, hits