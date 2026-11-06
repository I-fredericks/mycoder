#!/usr/bin/env python3
"""research.py: deep-research chat with your self-hosted model.

No tools, no code-editing harness — just a reasoning model you control.

Setup:
    export LLM_BASE_URL=https://code.pinkmailer.com/mycoder/v1
    export LLM_API_KEY=your-key
    export LLM_MODEL=coder

Run:
    python agent/research.py
    python agent/research.py "explain the trade-offs between ..."

Override the system prompt:
    python agent/research.py --system-prompt agent/prompts/research.txt
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PROMPT = Path(__file__).parent / "prompts" / "research.txt"


def chat(messages, base, key, model, max_tokens=2048, temperature=0.6):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="one-shot query; omit for interactive chat")
    ap.add_argument("--system-prompt", default=os.environ.get("LLM_SYSTEM_PROMPT", str(DEFAULT_PROMPT)),
                    help="path to system prompt file")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temp", type=float, default=0.6)
    args = ap.parse_args()

    base = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    key = os.environ.get("LLM_API_KEY", "none")
    model = os.environ.get("LLM_MODEL", "coder")

    system = Path(args.system_prompt).expanduser().read_text(encoding="utf-8")
    messages = [{"role": "system", "content": system}]

    if args.query:
        messages.append({"role": "user", "content": " ".join(args.query)})
        print(chat(messages, base, key, model, args.max_tokens, args.temp))
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
        messages.append({"role": "user", "content": line})
        answer = chat(messages, base, key, model, args.max_tokens, args.temp)
        print(f"\n{answer}\n")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
