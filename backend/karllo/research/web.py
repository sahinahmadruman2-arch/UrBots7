"""Optional web research. Never fabricates citations.

No search provider is bundled. If KARLLO_SEARCH_PROVIDER is unset, the API
reports that research is not configured. Credentials stay on the server.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse


def research_status() -> dict[str, Any]:
    provider = os.environ.get("KARLLO_SEARCH_PROVIDER", "").strip().lower()
    has_key = bool(os.environ.get("KARLLO_SEARCH_API_KEY", "").strip())
    configured = bool(provider) and (provider == "duckduckgo" or has_key)
    return {
        "configured": configured,
        "provider": provider or None,
        "note": (
            "Web research is optional and separate from model training. "
            "Karllo will not claim to browse the web unless a provider is configured."
            if not configured
            else f"Using provider '{provider}'. Results include source URLs."
        ),
    }


def search_web(query: str, k: int = 5) -> dict[str, Any]:
    status = research_status()
    if not status["configured"]:
        return {
            "ok": False,
            "configured": False,
            "results": [],
            "error": "No search provider configured. Set KARLLO_SEARCH_PROVIDER to enable web research.",
            "query": query,
        }
    provider = status["provider"]
    if provider == "duckduckgo":
        return _duckduckgo(query, k)
    return {
        "ok": False,
        "configured": True,
        "results": [],
        "error": f"Provider '{provider}' is not implemented in this build.",
        "query": query,
    }


def _duckduckgo(query: str, k: int) -> dict[str, Any]:
    """HTML instant-answer style lookup. May fail if outbound network is blocked."""
    try:
        import json
        from urllib.parse import urlencode
        from urllib.request import Request, urlopen

        url = "https://api.duckduckgo.com/?" + urlencode(
            {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        )
        req = Request(url, headers={"User-Agent": "KarlloAI/0.1"})
        with urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "configured": True,
            "results": [],
            "error": f"Search request failed: {exc}",
            "query": query,
        }
    results = []
    abstract = data.get("AbstractText") or ""
    abstract_url = data.get("AbstractURL") or data.get("AbstractSource") or ""
    if abstract:
        results.append(
            {
                "title": data.get("Heading") or query,
                "url": abstract_url,
                "snippet": abstract,
            }
        )
    for item in data.get("RelatedTopics", [])[:k]:
        if isinstance(item, dict) and item.get("Text"):
            results.append(
                {
                    "title": (item.get("Text") or "")[:80],
                    "url": item.get("FirstURL") or "",
                    "snippet": item.get("Text") or "",
                }
            )
    results = results[:k]
    # Drop results without a URL so we never invent citations.
    results = [r for r in results if r.get("url")]
    return {
        "ok": True,
        "configured": True,
        "results": results,
        "query": query,
        "note": "Retrieved facts are shown separately from Karllo's generated text.",
    }


def prefix_for_prompt(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    lines = ["Web research snippets (external; not model knowledge):"]
    for i, r in enumerate(results, start=1):
        host = urlparse(r.get("url") or "").netloc
        lines.append(f"[{i}] {r.get('title')} ({host}): {r.get('snippet', '')[:400]}")
        lines.append(f"    {r.get('url')}")
    return "\n".join(lines)
