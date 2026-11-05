"""Turn your own material into a fine-tuning dataset (chat JSONL).

Two sources, use either or both:

  1) Q&A pairs you wrote or curated, as JSONL lines like:
       {"prompt": "Write a Laravel middleware that ...", "response": "<code>"}
  2) Your own code repos: each source file becomes a "continue this file"
     example (first part is the prompt, the rest is the target). This teaches
     the model your style, naming and stack.

Usage:
  python prepare_data.py --pairs my_pairs.jsonl --repos ~/projects/velox ~/projects/other \
      --out data/train.jsonl --val data/val.jsonl

Output format (works directly with TRL's SFTTrainer):
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

SYSTEM = "You are a concise, expert coding assistant. Write correct, idiomatic code."

CODE_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".php", ".dart", ".java", ".kt",
    ".go", ".rs", ".c", ".cpp", ".h", ".cs", ".rb", ".sql", ".sh", ".html", ".css",
}
SKIP_DIRS = {
    ".git", "node_modules", "vendor", "build", "dist", ".dart_tool", "__pycache__",
    ".venv", "venv", ".idea", ".next", "storage",
}
SECRET_HINTS = ("password", "secret", "api_key", "apikey", "private_key", "token=")
MAX_FILE_CHARS = 12000
MIN_FILE_CHARS = 400


def msg(user, assistant):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def from_pairs(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("prompt") and row.get("response"):
                yield msg(row["prompt"], row["response"])


def looks_secret(text):
    low = text.lower()
    return any(h in low for h in SECRET_HINTS)


def from_repo(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() not in CODE_EXT:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if not (MIN_FILE_CHARS <= len(text) <= MAX_FILE_CHARS):
                continue
            if looks_secret(text):
                continue  # never train on files that may contain credentials
            lines = text.splitlines(keepends=True)
            cut = max(5, len(lines) // 3)
            head, tail = "".join(lines[:cut]), "".join(lines[cut:])
            rel = p.relative_to(root)
            user = (
                f"Continue this {p.suffix} file ({rel}). "
                f"Match the existing style.\n\n```\n{head}```"
            )
            yield msg(user, tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--repos", nargs="*", default=[])
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows, seen = [], set()
    sources = [from_pairs(p) for p in args.pairs] + [from_repo(Path(r)) for r in args.repos]
    for src in sources:
        for row in src:
            key = hashlib.md5(json.dumps(row, sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    if not rows:
        raise SystemExit("No examples found. Pass --pairs and/or --repos.")

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for path, data in ((args.out, train), (args.val, val)):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)}  val={len(val)}  -> {args.out}, {args.val}")


if __name__ == "__main__":
    main()
