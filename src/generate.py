# src/generate.py
import re
import time
import json
import requests
from typing import List, Dict, Tuple

"""
Local LLM generator using Ollama (mistral) running at http://localhost:11434.
No internet, no API keys, no quotas.

If Ollama is not running or model not pulled, we fall back to the extractive answer.
"""

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "mistral"            # you pulled this via: ollama pull mistral
MAX_TOKENS = 160                    # short, pin-point answers
TIMEOUT_SEC = 10                    # fail fast; we still have extractive fallback

PINPOINT_SYS = (
    "You are Aksum’s procurement assistant. "
    "Answer in 1–2 sentences ONLY, using only the provided snippets. "
    "Include numbers (percent, days, rupee amounts) if present. "
    "Cite sources inline like [1], [2] matching snippet order. "
    "If answer not in snippets, say: 'Not found in policy/contract.' "
    "Do NOT repeat long policy/FAQ blocks."
)

def _format_ctx(snippets: List[Dict], query: str) -> str:
    lines = []
    for i, s in enumerate(snippets[:3], 1):
        title = s.get("title", "")
        src = s.get("source", "")
        preview = (s.get("text_preview", "") or "").replace("\n", " ")
        preview = preview[:180]
        lines.append(f"[{i}] {title} | {src}\n{preview}\n")
    return "\n".join(lines)

def _fallback_extractive(answer_text: str) -> str:
    return answer_text

def _ollama_chat(model: str, system: str, user: str, timeout: int = TIMEOUT_SEC) -> str:
    """
    Minimal Ollama Chat API call. Returns the final message string or raises.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": MAX_TOKENS
        },
        "stream": False
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    content = (data.get("message", {}) or {}).get("content", "")
    return content.strip()

def generate_answer(query: str, snippets: List[Dict], extractive_answer: str) -> Tuple[str, bool]:
    """
    Try local LLM via Ollama; fall back to extractive if Ollama is unavailable or slow.
    """
    # quick precheck: if no snippets, return extractive (rag_chat handles this usually)
    if not snippets:
        return _fallback_extractive(extractive_answer), False

    # build a tight prompt
    ctx = _format_ctx(snippets, query)
    user_msg = (
        f"Question: {query}\n\n"
        f"Context snippets (use only these; cite as [1], [2]):\n{ctx}\n"
        "Answer now in 1–2 sentences with citations."
    )

    try:
        text = _ollama_chat(OLLAMA_MODEL, PINPOINT_SYS, user_msg, timeout=TIMEOUT_SEC)
        # tiny cleanup
        text = re.sub(r"\n{2,}", "\n", text).strip()
        if not text:
            return _fallback_extractive(extractive_answer), False
        return text, True
    except Exception as e:
        # common cases: Ollama not installed/running; model not pulled; timeout
        print("[generate] Ollama unavailable/failed → using extractive. Details:", e)
        return _fallback_extractive(extractive_answer), False
