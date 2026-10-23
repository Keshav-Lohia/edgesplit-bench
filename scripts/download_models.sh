#!/usr/bin/env bash
# Download the GGUF models used by the experiments into ./models
# If a URL 404s, open the Hugging Face repo page and copy the current file name into
# edgesplit/config.py (MODEL_PROFILES) - file names occasionally change.
set -euo pipefail
cd "$(dirname "$0")/../models"
get() { [ -f "$2" ] && echo "have $2" || curl -L --fail -o "$2" "https://huggingface.co/$1/resolve/main/$2"; }
get Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf
if [ "${1:-}" != "dev" ]; then
  get Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q2_k.gguf
  get Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf
  get Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q8_0.gguf
fi
ls -lh .
