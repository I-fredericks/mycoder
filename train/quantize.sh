#!/usr/bin/env bash
# Convert the merged HF model to a quantized GGUF for the 2GB server.
# Run on the GPU box / any machine with llama.cpp cloned and built.
#
#   bash quantize.sh out/merged mycoder-1.5b
#
# Then copy out/mycoder-1.5b-q4_k_m.gguf to the server as /opt/models/coder.gguf
# and run: sudo systemctl restart llama
set -euo pipefail

MERGED="${1:-out/merged}"
NAME="${2:-mycoder}"
LLAMA="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"

if [ ! -d "$LLAMA" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
fi
pip install -q -r "$LLAMA/requirements/requirements-convert_hf_to_gguf.txt"

# Build the quantize tool if needed
if [ ! -x "$LLAMA/build/bin/llama-quantize" ]; then
  cmake -S "$LLAMA" -B "$LLAMA/build" -DLLAMA_CURL=OFF
  cmake --build "$LLAMA/build" --config Release -j --target llama-quantize
fi

mkdir -p out
python "$LLAMA/convert_hf_to_gguf.py" "$MERGED" --outfile "out/$NAME-f16.gguf" --outtype f16
"$LLAMA/build/bin/llama-quantize" "out/$NAME-f16.gguf" "out/$NAME-q4_k_m.gguf" Q4_K_M
ls -lh "out/$NAME-q4_k_m.gguf"
