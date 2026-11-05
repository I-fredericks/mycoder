"""Generate fine-tuning pairs using a strong "teacher" API (Z.ai GLM, DeepSeek,
or any OpenAI-compatible endpoint). This is how you get Claude/GLM-level answers
in your training data without writing thousands of examples by hand.

Three modes, combinable:

  auto   invent diverse coding tasks for your stack, then answer each one
  tasks  answer tasks you wrote yourself (one per line in a text file)
  repos  read files from your own repos and create grounded Q&A pairs about
         that exact code (teaches the model your stack and style)

Keys come from the environment (GLM_API_KEY / ZAI_API_KEY, DEEPSEEK_API_KEY)
or CLI flags. Never put keys in files you commit.

Output is {"prompt": ..., "response": ...} JSONL, ready for:
  python prepare_data.py --pairs pairs.jsonl --repos ~/projects/x

Usage:
  export GLM_API_KEY=...
  export DEEPSEEK_API_KEY=...
  python generate_data.py --mode auto --stack "Python,FastAPI,SQL" --num 300
  python generate_data.py --mode tasks --tasks my_tasks.txt
  python generate_data.py --mode repos --repos ~/projects/velox --per-file 3
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

PROVIDERS = {
    "glm":      {"base": "https://api.z.ai/api/paas/v4", "model": "glm-4.5-flash"},
    "deepseek": {"base": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
}

CODE_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".php", ".dart", ".java", ".kt",
    ".go", ".rs", ".c", ".cpp", ".h", ".cs", ".rb", ".sql", ".sh", ".html", ".css",
}
SKIP_DIRS = {
    ".git", "node_modules", "vendor", "build", "dist", ".dart_tool", "__pycache__",
    ".venv", "venv", ".idea", ".next", "storage",
}
SECRET_HINTS = ("password", "secret", "api_key", "apikey", "private_key", "token=")
MAX_FILE_CHARS = 10000
MIN_FILE_CHARS = 300

AUTO_TOPICS = [
    "write a small utility function", "parse and transform text/JSON",
    "fix a subtle bug", "add input validation", "write a unit test",
    "optimize a slow function", "explain what code does step by step",
    "write a SQL query", "regex extraction", "handle errors properly",
    "write a CLI script", "refactor for readability",
]

JSON_ARRAY = re.compile(r"\[\s*\{.*\}\s*\]", re.S)


# ---------- teacher API ----------

def teacher_request(messages, provider, base=None, model=None, key=None,
                    max_tokens=2500, retries=3):
    p = PROVIDERS[provider]
    base = (base or os.environ.get(f"{provider.upper()}_BASE_URL") or p["base"]).rstrip("/")
    model = model or p["model"]
    key = key or os.environ.get(f"{provider.upper()}_API_KEY", "")
    if not key:
        raise SystemExit(f"No API key for {provider}. Set {provider.upper()}_API_KEY.")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.6}
    if provider == "glm":
        body["thinking"] = {"type": "disabled"}   # faster + cheaper for data gen
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (401, 403):
                raise SystemExit(f"{provider} rejected the API key (HTTP {e.code}).")
            if e.code == 429:
                wait = e.headers.get("Retry-After")
                time.sleep(float(wait) if wait and wait.replace(".", "").isdigit()
                           else min(60, 8 * (attempt + 1) ** 2))
                continue
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, KeyError,
                json.JSONDecodeError) as e:
            last = e
            time.sleep(2 * (attempt + 1))   # rate limits: back off and retry
    print(f"  !! {provider} failed after {retries} tries: {last}", file=sys.stderr)
    return None


# ---------- task invention (auto mode) ----------

def invent_tasks(n, stack, provider, seed):
    rng = random.Random(seed)
    topics = AUTO_TOPICS[:]
    rng.shuffle(topics)
    tasks = []
    while len(tasks) < n:
        want = min(20, n - len(tasks))
        picked = [topics[(i * 7 + len(tasks)) % len(topics)] for i in range(want)]
        prompt = (
            f"List {want} short, concrete coding tasks for a developer working with "
            f"{stack}. Theme each around one of: {', '.join(picked)}. "
            "One task per line, no numbering, no extra text. Each task must be "
            "solvable in under 60 lines of code and self-contained."
        )
        out = teacher_request([{"role": "user", "content": prompt}], provider)
        if not out:
            break
        for line in out.splitlines():
            line = line.strip().lstrip("0123456789.-) ")
            if 15 < len(line) < 300:
                tasks.append(line)
        if len(tasks) < n:
            time.sleep(1)
    return tasks[:n]


def answer_task(task, stack, provider):
    sys = ("You are an expert programmer. Write a correct, complete, production-quality "
           "answer: the code plus at most 2 short sentences of explanation. Use the "
           "user's stack. No preamble, no markdown headers, just a brief intro, the "
           "code in a fenced block, and a brief closing note if needed.")
    user = f"Stack: {stack}\n\nTask: {task}"
    return teacher_request(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        provider, max_tokens=2500)


# ---------- repo-grounded pairs ----------

def repo_files(root, max_files):
    from pathlib import Path
    files = []
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
            if any(h in text.lower() for h in SECRET_HINTS):
                continue  # never send possible secrets to an external API
            files.append((p, text))
    random.shuffle(files)
    return files[:max_files]


def pairs_from_file(path, text, k, provider):
    rel = path.name
    ext = path.suffix.lstrip(".") or "text"
    prompt = (
        f"You are creating training data for a coding assistant. Given the file below, "
        f"write {k} realistic developer instruction/answer pairs about THIS code.\n"
        "Rules:\n"
        "- Instructions a real developer would ask: fix/extend/refactor/explain/optimize.\n"
        "- Answers must be correct and complete enough to use directly (show the code).\n"
        "- Vary difficulty; include at least one explain and one small change.\n"
        '- Respond with ONLY a JSON array: [{"prompt": "...", "response": "..."}]\n\n'
        f"File: {rel}\n```{ext}\n{text}\n```"
    )
    out = teacher_request([{"role": "user", "content": prompt}], provider,
                          max_tokens=4000)
    if not out:
        return []
    m = JSON_ARRAY.search(out)
    if not m:
        return []
    try:
        rows = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [(r.get("prompt"), r.get("response")) for r in rows
            if isinstance(r, dict) and r.get("prompt") and r.get("response")]


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["auto", "tasks", "repos"], default="auto")
    ap.add_argument("--out", default="pairs.jsonl")
    ap.add_argument("--num", type=int, default=200, help="pairs to generate (auto mode)")
    ap.add_argument("--tasks", help="text file with one task per line (tasks mode)")
    ap.add_argument("--repos", nargs="*", default=[], help="repo dirs (repos mode)")
    ap.add_argument("--per-file", type=int, default=3, help="pairs per file (repos mode)")
    ap.add_argument("--max-files", type=int, default=100)
    ap.add_argument("--stack", default="Python", help="e.g. 'Python, FastAPI, React, SQL'")
    ap.add_argument("--provider", choices=list(PROVIDERS), default=None,
                    help="force one teacher; default alternates both")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    have = [p for p in PROVIDERS
            if os.environ.get(f"{p.upper()}_API_KEY") or os.environ.get(f"{p.upper()}_BASE_URL")]
    if args.provider:
        providers = [args.provider]
    elif have:
        providers = have
    else:
        raise SystemExit("No teacher keys found. Set GLM_API_KEY and/or DEEPSEEK_API_KEY.")

    jobs = []  # (description, kind, payload)
    if args.mode == "auto":
        print(f"Inventing {args.num} tasks with {providers[0]}...")
        for t in invent_tasks(args.num, args.stack, providers[0], args.seed):
            jobs.append((t, "answer", t))
    elif args.mode == "tasks":
        if not args.tasks:
            raise SystemExit("--mode tasks needs --tasks file")
        with open(args.tasks, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 10:
                    jobs.append((line, "answer", line))
    elif args.mode == "repos":
        if not args.repos:
            raise SystemExit("--mode repos needs --repos dirs")
        for r in args.repos:
            files = repo_files(r, args.max_files)
            print(f"{r}: {len(files)} files eligible")
            for p, text in files:
                jobs.append((str(p), "file", (p, text)))

    print(f"{len(jobs)} jobs, teachers: {', '.join(providers)}, workers={args.workers}")

    results, seen, failed = [], set(), 0
    rot = iter(range(len(jobs) * 10))  # round-robin per job, independent of success

    def pick_provider():
        return providers[next(rot) % len(providers)]

    def run(job):
        desc, kind, payload = job
        prov = pick_provider()
        if kind == "answer":
            resp = answer_task(payload, args.stack, prov)
            return (payload, resp) if resp else None
        p, text = payload
        return pairs_from_file(p, text, args.per_file, prov)

    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run, j): j for j in jobs}
        done = 0
        for fut in cf.as_completed(futs):
            done += 1
            desc = futs[fut][0]
            try:
                got = fut.result()
            except SystemExit as e:
                raise e
            except Exception as e:
                got, failed = None, failed + 1
                print(f"  !! {desc}: {type(e).__name__}: {e}", file=sys.stderr)
            rows = []
            if got:
                if isinstance(got, tuple):
                    rows = [got]
                else:
                    rows = got
            for prompt, response in rows:
                key = re.sub(r"\s+", " ", prompt).strip().lower()
                if key in seen or len(response) < 40:
                    continue
                seen.add(key)
                results.append({"prompt": prompt, "response": response})
            if done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} jobs done, {len(results)} unique pairs")

    rng.shuffle(results)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} pairs to {args.out}  (failed jobs: {failed})")
    print("Next: python prepare_data.py --pairs " + args.out +
          " --repos <your repos> --out data/train.jsonl --val data/val.jsonl")


if __name__ == "__main__":
    main()
