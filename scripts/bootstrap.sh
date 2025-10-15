#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_DIR="$PROJECT_ROOT/assets"
LLM_DIR="$ASSETS_DIR/llm/LiquidAI/LFM2-350M-GGUF"
TTS_DIR="$ASSETS_DIR/tts/kitten-tts"

python_bin=${PYTHON:-python3}

if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry not found. Installing locally..."
  curl -sSL https://install.python-poetry.org | $python_bin -
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$PROJECT_ROOT"
poetry env use "$python_bin"
poetry install --with dev

mkdir -p "$LLM_DIR" "$TTS_DIR"

LLM_FILE="$LLM_DIR/LFM2-350M-Q4_K_M.gguf"
if [ ! -f "$LLM_FILE" ]; then
  echo "Downloading LFM2-350M GGUF (Q4_K_M)..."
  curl -L "https://huggingface.co/LiquidAI/LFM2-350M-GGUF/resolve/main/LFM2-350M-Q4_K_M.gguf" -o "$LLM_FILE"
fi

if [ ! -f "$TTS_DIR/kitten_tts_nano_v0_2.onnx" ]; then
  poetry run voice-agent models pull tts/kitten-nano-0.2 --dest "$TTS_DIR"
fi

echo "Bootstrap complete. Activate the environment with:"
echo "  poetry shell"
