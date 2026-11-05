# mycoder: your own coding AI

Three parts, built to fit a **2GB RAM / 40GB SSD** server:

```
server/   host the model (llama.cpp + nginx + API key)
train/    customize a model on your own code (GPU needed, done off-server)
agent/    Claude Code-style terminal agent that talks to your model
```

## Be realistic first

Claude Code and Z.ai's GLM are **harness + giant model**. The harness (agent loop, tools, file edits) you can build fully, and `agent/agent.py` is a working one. The giant model (hundreds of billions of parameters, trained on thousands of GPUs) you cannot train or host yourself. What you can do:

| Goal | How |
|---|---|
| Private, always-on coding model on your server | Fine-tuned 0.5B-1.5B model, 4-bit GGUF, runs on CPU in <1.2GB RAM |
| Quality closer to Claude/GLM for hard tasks | Point the same agent at a hosted API (set `LLM_BASE_URL`), keep your server model for cheap/private work |
| Model that knows *your* stack and style | Fine-tune on your repos (steps below) |

Expect a 1.5B model to handle autocomplete, small functions, explaining code and simple edits. It will struggle with multi-file refactors and reliable tool calling. If that bites, the best upgrade is a bigger server (8GB+ RAM to run a 7B) rather than more training.

## Roadmap

**Phase 1: get a model running (1 evening)**
1. `sudo bash server/setup.sh` on the server (adds swap, builds llama.cpp, downloads Qwen2.5-Coder 1.5B Q4, starts a systemd service on 127.0.0.1:8080).
2. `curl http://127.0.0.1:8080/v1/models` to confirm it responds.
3. Put `server/nginx.conf` in `/etc/nginx/conf.d/`, set your API key and domain, run certbot for HTTPS.

**Phase 2: use it with the agent (same evening)**
```bash
export LLM_BASE_URL=https://code.example.com/v1
export LLM_API_KEY=your-key
cd ~/your-project
python /path/to/agent/agent.py "explain what main.py does"
```
It asks before writing files or running commands (`--yes` skips that). Editors: the **Continue** extension in VS Code works with the same endpoint.

**Phase 3: personalize the model (a weekend)**
1. Generate training pairs with your GLM/DeepSeek keys (distillation):
   ```bash
   export GLM_API_KEY=... DEEPSEEK_API_KEY=...
   # auto-invent tasks for your stack:
   python train/generate_data.py --mode auto --stack "Python,FastAPI,SQL" --num 300
   # and/or grounded pairs from your real repos:
   python train/generate_data.py --mode repos --repos ~/projects/velox --per-file 3
   ```
   Files that look like they contain secrets are never sent to the teacher API.
2. Mix in anything you wrote by hand:
   `python train/prepare_data.py --repos ~/projects/velox ~/projects/other --pairs pairs.jsonl`
   Add hand-written prompt/response pairs for the things you ask most. 500 to 5,000 good examples is enough to see an effect.
3. Fine-tune on a free Colab/Kaggle T4 or a rented GPU (RunPod/Vast.ai, a few dollars):
   `python train/finetune.py --train data/train.jsonl --val data/val.jsonl`
4. Merge and quantize:
   `python train/merge_lora.py` then `bash train/quantize.sh out/merged mycoder-1.5b`
5. `scp out/mycoder-1.5b-q4_k_m.gguf server:/opt/models/coder.gguf` and `sudo systemctl restart llama`.

**Phase 4: improve**
- Keep a log of prompts where the model fails, fix the answers by hand, add them to your pairs file, and retrain. This loop is what actually improves a personal model.
- Evaluate with a small fixed set of your own tasks before and after each training run, so you know whether it got better.
- Later: swap in a newer small coder base model (`--base` in the scripts), or a 7B once you have more RAM.

## Things to know

- **Fine-tuning scripts weren't run here** (no GPU in this environment). `prepare_data.py` and the agent loop were tested; `finetune.py` follows current TRL/PEFT APIs, but those change often, so if an argument errors, check the docs for your installed version.
- **Tool calling on small models is flaky.** The agent handles both native tool calls and `<tool_call>` JSON in plain text. If the model loops or makes bad edits, shorten the task or try a hosted model for that job.
- **CPU speed:** expect a few tokens per second on 2GB hardware. Keep context at 4096 or lower.
- **Licenses:** check the license of any base model you fine-tune (Qwen2.5-Coder 1.5B is Apache 2.0; some sizes differ).
- **Security:** never expose port 8080 directly, and don't train on files containing credentials. API keys belong in environment variables (or an uncommitted `.env`), never in committed files or shell history — if a key was ever pasted into chat or a screenshot, rotate it.
