#!/usr/bin/env python3
"""mycoder: a small Claude Code-style terminal agent for any OpenAI-compatible
model endpoint (your llama.cpp server, Ollama, or a hosted API like Z.ai / Qwen).

No third-party dependencies. Python 3.9+.

Setup:
    export LLM_BASE_URL=https://code.example.com/v1     # or http://127.0.0.1:8080/v1
    export LLM_API_KEY=your-key
    export LLM_MODEL=coder                              # llama.cpp ignores the name

Run inside the project you want it to work on:
    python agent.py                    # interactive chat
    python agent.py "add input validation to signup.py"
    python agent.py --yes "run the tests and fix failures"   # skip confirmations

Customize behavior and guardrails:
    python agent.py --system-prompt agent/my-prompt.txt
    python agent.py --rules agent/my-rules.txt

File tools are confined to the current directory. Writes and shell commands ask
for confirmation unless you pass --yes.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path.cwd().resolve()
MAX_OUTPUT = 6000          # chars returned to the model per tool call (keep context small)
MAX_STEPS = 20             # tool-use rounds per user request
AUTO_YES = False
CUSTOM_RULES = []

DEFAULT_SYSTEM_PROMPT = f"""You are mycoder, a terminal coding agent working in the project at {ROOT}.
Use the tools to inspect and change files, then give a short final answer.
Rules:
- Read a file before editing it. Prefer edit_file over rewriting whole files.
- Keep changes minimal and match the existing style.
- After changes, run tests or a syntax check with run_command when possible.
- Be concise. When the task is done, reply with a brief summary and no tool call."""

TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir", "description": "List files in a directory (relative path).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file with line numbers.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start": {"type": "integer", "description": "first line, 1-based"},
            "end": {"type": "integer", "description": "last line"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a file with full content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace one exact occurrence of old_text with new_text.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {
        "name": "search", "description": "Regex search across project files. Returns file:line: text.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the project directory (60s timeout).",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

SKIP_DIRS = {".git", "node_modules", "vendor", "__pycache__", ".venv", "venv", "build", "dist"}


# ---------- helpers ----------

def load_rules(path):
    """Load a list of policy rules from a text file. Empty lines and # comments ignored."""
    p = Path(path).expanduser()
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")]


def input_guard(text):
    """Block user inputs that violate your custom rules. Returns (ok, reason)."""
    lowered = text.lower()
    for rule in CUSTOM_RULES:
        if rule.startswith("BLOCK:"):
            _, needle = rule.split(":", 1)
            if needle.strip().lower() in lowered:
                return False, f"Blocked by local policy: {needle.strip()}"
    return True, None


def output_guard(text):
    """Block model outputs that violate your custom rules. Returns (ok, reason)."""
    lowered = text.lower()
    for rule in CUSTOM_RULES:
        if rule.startswith("OUTPUT_BLOCK:"):
            _, needle = rule.split(":", 1)
            if needle.strip().lower() in lowered:
                return False, f"Output blocked by local policy: {needle.strip()}"
    return True, None


def safe_path(rel):
    p = (ROOT / (rel or ".")).resolve()
    if p != ROOT and ROOT not in p.parents:
        raise ValueError(f"path escapes project directory: {rel}")
    return p


def clip(text):
    if len(text) <= MAX_OUTPUT:
        return text
    return text[:MAX_OUTPUT] + f"\n... [truncated {len(text) - MAX_OUTPUT} chars]"


def confirm(prompt):
    if AUTO_YES:
        return True
    try:
        return input(f"\033[33m{prompt} [y/N] \033[0m").strip().lower() in ("y", "yes")
    except EOFError:
        return False


# ---------- tools ----------

def t_list_dir(path="."):
    p = safe_path(path)
    out = []
    for e in sorted(p.iterdir()):
        if e.name in SKIP_DIRS:
            continue
        out.append(e.name + ("/" if e.is_dir() else ""))
    return "\n".join(out) or "(empty)"


def t_read_file(path, start=None, end=None):
    lines = safe_path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    s = max(1, int(start or 1))
    e = min(len(lines), int(end or len(lines)))
    return clip("\n".join(f"{i}: {lines[i - 1]}" for i in range(s, e + 1)))


def t_write_file(path, content):
    p = safe_path(path)
    if not confirm(f"Write {path} ({len(content)} chars)?"):
        return "User declined."
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {path}"


def t_edit_file(path, old_text, new_text):
    p = safe_path(path)
    text = p.read_text(encoding="utf-8")
    n = text.count(old_text)
    if n == 0:
        return "Error: old_text not found. Re-read the file and copy the text exactly."
    if n > 1:
        return f"Error: old_text matches {n} places. Include more surrounding lines."
    if not confirm(f"Edit {path}?\n--- old\n{old_text}\n+++ new\n{new_text}\n"):
        return "User declined."
    p.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {path}"


def t_search(pattern, path="."):
    rx = re.compile(pattern)
    base = safe_path(path)
    hits = []
    files = [base] if base.is_file() else (
        f for f in base.rglob("*") if f.is_file() and not (set(f.parts) & SKIP_DIRS))
    for f in files:
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()[:200]}")
                    if len(hits) >= 80:
                        return "\n".join(hits) + "\n... (more matches)"
        except (UnicodeDecodeError, OSError):
            continue
    return "\n".join(hits) or "No matches."


def t_run_command(command):
    if not confirm(f"Run: {command}"):
        return "User declined."
    try:
        r = subprocess.run(command, shell=True, cwd=ROOT, capture_output=True,
                           text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 60s."
    return clip(f"exit={r.returncode}\n{r.stdout}{r.stderr}")


DISPATCH = {
    "list_dir": t_list_dir, "read_file": t_read_file, "write_file": t_write_file,
    "edit_file": t_edit_file, "search": t_search, "run_command": t_run_command,
}


def run_tool(name, args):
    fn = DISPATCH.get(name)
    if not fn:
        return f"Error: unknown tool {name}"
    try:
        return fn(**args)
    except TypeError as e:
        return f"Error: bad arguments for {name}: {e}"
    except Exception as e:  # keep the loop alive; let the model see the error
        return f"Error: {type(e).__name__}: {e}"


# ---------- model call ----------

def chat(messages):
    base = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
    body = {
        "model": os.environ.get("LLM_MODEL", "coder"),
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'none')}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.load(resp)["choices"][0]["message"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Model server error {e.code}: {e.read().decode()[:500]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Cannot reach model at {base}: {e.reason}")


TOOL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
BARE_TOOL = re.compile(r'\{[^}]*"name"[^}]*"arguments"[^}]*\}', re.S)


def _bare_tool_calls(content):
    """Some small models emit tool calls as plain JSON blocks without tags."""
    calls = []
    for m in BARE_TOOL.finditer(content):
        # expand to balanced braces
        start = m.start()
        depth = 0
        end = None
        for i, ch in enumerate(content[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        try:
            obj = json.loads(content[start:end])
            if "name" in obj and "arguments" in obj:
                calls.append((f"bare{len(calls)}", obj["name"], obj.get("arguments", {})))
        except json.JSONDecodeError:
            continue
    return calls


def extract_calls(msg):
    """Native tool_calls first; fall back to <tool_call>{json}</tool_call> in text,
    which small models sometimes emit when the server doesn't parse it."""
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc["function"]
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        calls.append((tc.get("id", fn["name"]), fn["name"], args))
    if not calls and msg.get("content"):
        for i, m in enumerate(TOOL_TAG.finditer(msg["content"])):
            try:
                obj = json.loads(m.group(1))
                calls.append((f"fallback{i}", obj["name"], obj.get("arguments", {})))
            except (json.JSONDecodeError, KeyError):
                continue
        if not calls:
            calls = _bare_tool_calls(msg["content"])
    return calls


def run_turn(messages):
    for _ in range(MAX_STEPS):
        msg = chat(messages)
        content = msg.get("content", "")
        ok, reason = output_guard(content)
        if not ok:
            print(f"\n[Policy block] {reason}\n")
            return
        calls = extract_calls(msg)
        messages.append({k: v for k, v in msg.items() if v is not None})
        if not calls:
            print(f"\n{content.strip()}\n")
            return
        for call_id, name, args in calls:
            print(f"\033[36m> {name}({json.dumps(args)[:150]})\033[0m")
            result = run_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
    print("Stopped: reached the step limit for this request.")


def main():
    global AUTO_YES, CUSTOM_RULES
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="*", help="one-shot task; omit for interactive mode")
    ap.add_argument("--yes", action="store_true", help="auto-approve writes and commands")
    ap.add_argument("--system-prompt", help="path to a custom system prompt file")
    ap.add_argument("--rules", help="path to a custom rules/policy file")
    args = ap.parse_args()
    AUTO_YES = args.yes
    CUSTOM_RULES = load_rules(args.rules) if args.rules else []

    system = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt:
        system = Path(args.system_prompt).expanduser().read_text(encoding="utf-8")
    elif os.environ.get("LLM_SYSTEM_PROMPT"):
        system = Path(os.environ["LLM_SYSTEM_PROMPT"]).expanduser().read_text(encoding="utf-8")

    messages = [{"role": "system", "content": system}]
    if args.task:
        user_text = " ".join(args.task)
        ok, reason = input_guard(user_text)
        if not ok:
            print(f"[Policy block] {reason}")
            return
        messages.append({"role": "user", "content": user_text})
        run_turn(messages)
        return

    print(f"mycoder in {ROOT}  (Ctrl-D to quit)")
    while True:
        try:
            line = input("\n\033[32m> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        ok, reason = input_guard(line)
        if not ok:
            print(f"[Policy block] {reason}")
            continue
        messages.append({"role": "user", "content": line})
        run_turn(messages)


if __name__ == "__main__":
    main()
