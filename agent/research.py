#!/usr/bin/env python3
"""research.py: deep-research assistant with automatic web search.

The script decides when to search the web and fetches the top result, then asks
the self-hosted model to reason over the combined context.

Setup:
    export LLM_BASE_URL=https://code.pinkmailer.com/mycoder/v1
    export LLM_API_KEY=your-key
    export LLM_MODEL=coder

Run:
    python agent/research.py
    python agent/research.py "how do I implement JWT auth in Express?"
    python agent/research.py --fetch-url https://expressjs.com/en/guide/routing.html "explain express routing"
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import web_tools

PROMPT = Path(__file__).parent / "prompts" / "research.txt"


def chat(messages, base, key, model, max_tokens=2048, temperature=0.4):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Model error {e.code}: {e.read().decode()[:500]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Cannot reach model at {base}: {e.reason}")


def wants_web_search(text):
    """Heuristic: user explicitly asks for web search or uses recent-web terms."""
    t = text.lower()
    web_terms = [
        "latest", "current", "recent", "news", "version", "release",
        "search the web", "look up", "find online", "documentation",
        "how to", "tutorial", "example of",
    ]
    return any(term in t for term in web_terms)


def research(query, base, key, model, max_tokens=2048, fetch_url=None, results=3):
    system = PROMPT.read_text(encoding="utf-8")
    context = []

    if fetch_url:
        print("> fetching page...")
        context.append(f"Fetched page {fetch_url}:\n" + web_tools.fetch_page(fetch_url))
    elif wants_web_search(query):
        print("> searching the web...")
        hits = web_tools.search_web(query, n=results)
        context.append("Web search results:\n" + json.dumps(hits, indent=2))
        if hits and hits[0].get("url"):
            print("> fetching top result...")
            context.append("Top result content:\n" + web_tools.fetch_page(hits[0]["url"]))

    user_msg = query
    if context:
        user_msg = (
            f"User question: {query}\n\n"
            "Use the following web context to answer. Cite sources when possible.\n\n" +
            "\n\n---\n\n".join(context)
        )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    return chat(messages, base, key, model, max_tokens)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="one-shot query; omit for interactive chat")
    ap.add_argument("--fetch-url", help="fetch this URL instead of searching")
    ap.add_argument("--results", type=int, default=3, help="number of web results to include")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temp", type=float, default=0.4)
    args = ap.parse_args()

    base = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    key = os.environ.get("LLM_API_KEY", "none")
    model = os.environ.get("LLM_MODEL", "coder")

    if args.query:
        q = " ".join(args.query)
        print(research(q, base, key, model, args.max_tokens, args.fetch_url, args.results))
        return

    print("Research assistant (Ctrl-D to quit)")
    while True:
        try:
            line = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        print(research(line, base, key, model, args.max_tokens, args.fetch_url, args.results))


if __name__ == "__main__":
    main()
