#!/usr/bin/env bash
# One-shot setup for a 2GB RAM / 40GB SSD Ubuntu/Debian server.
# Installs llama.cpp (CPU), adds swap, downloads a small quantized coder model,
# and runs it as a systemd service bound to localhost only.
#
# Usage:  sudo bash setup.sh
set -euo pipefail

MODEL_URL="${MODEL_URL:-https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"
MODEL_DIR=/opt/models
MODEL_FILE="$MODEL_DIR/coder.gguf"
CTX="${CTX:-8192}"          # context window; raise only if RAM allows (this VPS uses 16384)
THREADS="$(nproc)"

echo "==> Installing build tools"
apt-get update
apt-get install -y git build-essential cmake curl nginx apache2-utils

echo "==> Creating 4GB swap (skipped if it exists)"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 4G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> Building llama.cpp"
if [ ! -d /opt/llama.cpp ]; then
  git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp
fi
cd /opt/llama.cpp
git pull --ff-only || true
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j"$THREADS" --target llama-server

echo "==> Downloading model"
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_FILE" ]; then
  curl -L --fail -o "$MODEL_FILE" "$MODEL_URL"
fi

echo "==> Installing systemd service"
cat > /etc/systemd/system/llama.service <<EOF
[Unit]
Description=llama.cpp server (personal coding model)
After=network.target

[Service]
ExecStart=/opt/llama.cpp/build/bin/llama-server \\
  -m $MODEL_FILE \\
  --host 127.0.0.1 --port 8080 \\
  -c $CTX -t $THREADS \\
  --jinja
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now llama.service

echo "==> Done. Test locally with:"
echo "curl http://127.0.0.1:8080/v1/models"
echo "Next: set up nginx + API key (see server/nginx.conf and README)."
