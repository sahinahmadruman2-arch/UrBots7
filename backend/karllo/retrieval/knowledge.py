"""Document knowledge base with a labeled BM25 lexical search fallback.

This module never updates model weights. Retrieval is optional and independent
of training. Semantic embeddings are used only when a compatible local model
can be imported; otherwise BM25 is used and clearly labeled.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from karllo.paths import KNOWLEDGE_DIR, ensure_dirs
from karllo.training.data import parse_qa_book

TOKEN_RE = re.compile(r"[a-z0-9]+")
MAX_DOC_BYTES = 5 * 1024 * 1024


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + size]
        chunks.append(chunk.strip())
        i += size - overlap
        if i >= len(text):
            break
    return [c for c in chunks if c]


def embedding_status() -> dict[str, Any]:
    try:
        import importlib

        importlib.import_module("sentence_transformers")
        return {
            "available": True,
            "method": "sentence-transformers",
            "label": "Semantic embeddings",
        }
    except Exception:
        return {
            "available": False,
            "method": "bm25",
            "label": "Basic BM25 lexical search (no embedding model installed)",
        }


class KnowledgeBase:
    def __init__(self) -> None:
        ensure_dirs()
        self.root = KNOWLEDGE_DIR
        self._docs_index = self.root / "index.json"

    def _load_index(self) -> list[dict[str, Any]]:
        if not self._docs_index.exists():
            return []
        return json.loads(self._docs_index.read_text(encoding="utf-8"))

    def _save_index(self, docs: list[dict[str, Any]]) -> None:
        self._docs_index.write_text(json.dumps(docs, indent=2), encoding="utf-8")

    def list_documents(self) -> list[dict[str, Any]]:
        return self._load_index()

    def add_document(self, name: str, text: str, source: str = "upload") -> dict[str, Any]:
        if len(text.encode("utf-8")) > MAX_DOC_BYTES:
            raise ValueError("Document exceeds 5 MB limit")
        doc_id = uuid.uuid4().hex[:12]
        dest = self.root / doc_id
        dest.mkdir(parents=True, exist_ok=True)
        chunks: list[dict[str, Any]] = []
        qa = parse_qa_book(text) if "Question:" in text and "Answer:" in text else []
        if qa:
            for i, ex in enumerate(qa):
                body = f"Question: {ex['prompt']}\nAnswer: {ex['response']}"
                chunks.append({"id": f"{doc_id}-{i}", "text": body, "kind": "qa"})
        else:
            for i, ch in enumerate(_chunk_text(text)):
                chunks.append({"id": f"{doc_id}-{i}", "text": ch, "kind": "passage"})
        (dest / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        (dest / "raw.txt").write_text(text, encoding="utf-8")
        rec = {
            "id": doc_id,
            "name": name,
            "source": source,
            "n_chunks": len(chunks),
            "chars": len(text),
            "created_at": time.time(),
            "enabled": True,
        }
        docs = self._load_index()
        docs.append(rec)
        self._save_index(docs)
        return rec

    def delete_document(self, doc_id: str) -> None:
        if "/" in doc_id or ".." in doc_id:
            raise ValueError("invalid id")
        docs = [d for d in self._load_index() if d["id"] != doc_id]
        self._save_index(docs)
        dest = self.root / doc_id
        if dest.exists():
            import shutil

            shutil.rmtree(dest)

    def _all_chunks(self) -> list[dict[str, Any]]:
        out = []
        for doc in self._load_index():
            if not doc.get("enabled", True):
                continue
            path = self.root / doc["id"] / "chunks.json"
            if not path.exists():
                continue
            for ch in json.loads(path.read_text(encoding="utf-8")):
                ch["doc_id"] = doc["id"]
                ch["doc_name"] = doc["name"]
                out.append(ch)
        return out

    def search(self, query: str, k: int = 4) -> dict[str, Any]:
        method = embedding_status()
        chunks = self._all_chunks()
        if not chunks:
            return {"hits": [], "method": method, "query": query}
        hits = _bm25(query, chunks, k=k)
        return {
            "hits": hits,
            "method": method,
            "query": query,
            "note": "Retrieval does not train or update Karllo model weights.",
        }

    def prefix_for_prompt(self, query: str, k: int = 4) -> tuple[str, list[dict[str, Any]]]:
        result = self.search(query, k=k)
        hits = result["hits"]
        if not hits:
            return "", []
        lines = [
            "Retrieved knowledge passages (not model weights). Cite these when relevant:"
        ]
        for i, h in enumerate(hits, start=1):
            lines.append(f"[{i}] ({h['doc_name']}) {h['text'][:500]}")
        return "\n".join(lines), hits


def _bm25(query: str, chunks: list[dict[str, Any]], k: int, k1: float = 1.5, b: float = 0.75) -> list[dict[str, Any]]:
    q = _tokens(query)
    if not q:
        return []
    docs_toks = [_tokens(c["text"]) for c in chunks]
    avgdl = sum(len(t) for t in docs_toks) / max(1, len(docs_toks))
    df: Counter = Counter()
    for toks in docs_toks:
        df.update(set(toks))
    n = len(chunks)
    scored = []
    for chunk, toks in zip(chunks, docs_toks):
        tf = Counter(toks)
        dl = len(toks) or 1
        score = 0.0
        for term in q:
            if tf[term] == 0:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            denom = tf[term] + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf[term] * (k1 + 1) / denom)
        if score > 0:
            scored.append({**chunk, "score": round(score, 4)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]
