#!/usr/bin/env python3
"""OpenAI-compatible proxy that adds web search to a self-hosted model.

Runs between Kilo Code and llama.cpp. For every chat request it checks whether
the last user message needs live web info; if so, it searches DuckDuckGo,
fetches the top result, and prepends the context before forwarding to the model.

No third-party dependencies.
"""
import http.client
import html
import json
import re
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import quote

UPSTREAM = "http://127.0.0.1:8080"   # llama.cpp server
HOST = "127.0.0.1"
PORT = 8081

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
MAX_PAGE_CHARS = 10000
MAX_WEB_CHARS = 12000


def _get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"HTTP error {e.code}"
    except Exception as e:
        return f"Fetch error: {type(e).__name__}: {e}"


# ---------- DuckDuckGo search ----------

class _DDGParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self._in_result = False
        self._in_link = False
        self._in_snippet = False
        self._current = {}
        self._href = None
        self._text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        cls = attrs.get("class", "")
        if tag == "div" and "result" in cls and "result__a" not in cls:
            self._in_result = True
            self._current = {}
        if self._in_result and tag == "a" and "result__a" in cls:
            self._in_link = True
            self._href = attrs.get("href", "")
            self._text = []
        if self._in_result and tag == "a" and "result__snippet" in cls:
            self._in_snippet = True
            self._text = []

    def handle_endtag(self, tag):
        if self._in_link and tag == "a":
            self._current["title"] = html.unescape("".join(self._text)).strip()
            self._current["url"] = self._href
            self._in_link = False
        if self._in_snippet and tag == "a":
            self._current["snippet"] = html.unescape("".join(self._text)).strip()
            self._in_snippet = False
            if self._current:
                self.results.append(self._current)
                self._current = {}
            self._in_result = False

    def handle_data(self, data):
        if self._in_link or self._in_snippet:
            self._text.append(data)


def search_web(query, n=3):
    url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
    text = _get(url, timeout=25)
    if text.startswith(("HTTP error", "Fetch error")):
        return [{"title": "Search failed", "url": "", "snippet": text}]
    parser = _DDGParser()
    parser.feed(text)
    return parser.results[:n]


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "header", "aside"):
            self._skip += 1
        elif tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "li", "tr", "pre"):
            self.text.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header", "aside"):
            self._skip -= 1
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "li", "tr", "pre"):
            self.text.append("\n")

    def handle_data(self, data):
        if self._skip <= 0:
            self.text.append(data)


def fetch_page(url):
    if not url:
        return "No URL provided."
    text = _get(url, timeout=30)
    if text.startswith(("HTTP error", "Fetch error")):
        return text
    parser = _TextExtractor()
    try:
        parser.feed(text)
    except Exception:
        pass
    body = html.unescape(re.sub(r"\n\s*\n+", "\n\n", "".join(parser.text)))
    body = re.sub(r"[ \t]+", " ", body)
    if len(body) > MAX_PAGE_CHARS:
        body = body[:MAX_PAGE_CHARS] + f"\n... [truncated {len(body) - MAX_PAGE_CHARS} chars]"
    return body.strip() or "(no text extracted)"


# ---------- proxy logic ----------

def wants_web_search(text):
    t = text.lower()
    web_terms = [
        "latest", "current", "recent", "news", "version", "release",
        "search the web", "look up", "find online", "documentation",
        "how to", "tutorial", "example of",
    ]
    return any(term in t for term in web_terms)


def build_web_context(query):
    hits = search_web(query, n=3)
    parts = ["Web search results:"]
    for i, h in enumerate(hits, 1):
        parts.append(f"{i}. {h.get('title', '')}\n   URL: {h.get('url', '')}\n   {h.get('snippet', '')}")
    if hits and hits[0].get("url"):
        parts.append("\nFetched content from top result:\n" + fetch_page(hits[0]["url"]))
    ctx = "\n\n".join(parts)
    if len(ctx) > MAX_WEB_CHARS:
        ctx = ctx[:MAX_WEB_CHARS] + f"\n... [truncated {len(ctx) - MAX_WEB_CHARS} chars]"
    return ctx


def forward(path, body=None, method="GET"):
    """Forward request to upstream llama.cpp server."""
    url = f"{UPSTREAM.rstrip('/')}{path}"
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"User-Agent": USER_AGENT,
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _send(self, code, body, headers=None):
        headers = headers or {}
        self.send_response(code)
        self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
        for k, v in headers.items():
            if k.lower() not in ("content-type", "transfer-encoding", "content-length"):
                self.send_header(k, v)
        data = body if isinstance(body, bytes) else body.encode()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            code, body, hdrs = forward("/v1/models")
            self._send(code, body, hdrs)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Bad JSON")
            return

        messages = req.get("messages", [])
        last_user = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break

        if last_user and wants_web_search(last_user):
            try:
                ctx = build_web_context(last_user)
                ctx_msg = {"role": "system", "content": f"Web research context:\n{ctx}"}
                if messages and messages[0].get("role") == "system":
                    messages.insert(1, ctx_msg)
                else:
                    messages.insert(0, ctx_msg)
                req["messages"] = messages
            except Exception as e:
                sys.stderr.write(f"web search failed: {e}\n")

        code, resp_body, hdrs = forward("/v1/chat/completions",
                                        body=json.dumps(req).encode(),
                                        method="POST")
        self._send(code, resp_body, hdrs)


def main():
    server = HTTPServer((HOST, PORT), Handler)
    print(f"mycoder research proxy on {HOST}:{PORT} → {UPSTREAM}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
