#!/usr/bin/env bash
set -euo pipefail

# Setup script for QTIP 1MAD environment
# Tested on: Ubuntu 22.04, Python 3.10, CUDA 12.6, RTX 3090
#
# Usage:
#   ./setup_env.sh [ENV_DIR]
#
# Example:
#   ./setup_env.sh              # creates ../.env-qtip
#   ./setup_env.sh /path/to/env # creates env at specified path

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${1:-$(dirname "$SCRIPT_DIR")/.env-qtip}"
FHT_DIR="$(dirname "$SCRIPT_DIR")/fast-hadamard-transform"

echo "[1/6] Creating virtual environment at $ENV_DIR"
if [[ -d "$ENV_DIR" ]]; then
    echo "  Environment already exists, skipping creation."
else
    python3 -m venv "$ENV_DIR"
fi

PY="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"

echo "[2/6] Upgrading pip"
"$PIP" install --upgrade pip setuptools wheel

echo "[3/6] Installing PyTorch 2.4.0 (CUDA 12.1)"
"$PIP" install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

echo "[4/6] Installing Python dependencies"
"$PIP" install \
    accelerate>=0.34.2 \
    cuda-python==12.6.0 \
    datasets>=2.20.0 \
    glog==0.3.1 \
    huggingface-hub==0.24.0 \
    lm_eval>=0.4.4 \
    numpy==2.1.2 \
    scipy==1.14.1 \
    tqdm==4.66.4 \
    transformers==4.45.2 \
    sentencepiece \
    protobuf

echo "[5/6] Installing fast_hadamard_transform from source"
if [[ -d "$FHT_DIR" ]]; then
    echo "  Found existing source at $FHT_DIR"
else
    echo "  Cloning fast-hadamard-transform..."
    git clone https://github.com/Dao-AILab/fast-hadamard-transform.git "$FHT_DIR"
fi
"$PIP" install -e "$FHT_DIR"

echo "[6/6] Installing QTIP CUDA kernels"
"$PIP" install -e "$SCRIPT_DIR/qtip-kernels"

echo ""
echo "===== Setup complete ====="
echo "Activate with: source $ENV_DIR/bin/activate"
echo "Run pipeline:  ./run_qtip_1mad_pipeline.sh <MODEL_ID> <RUN_TAG>"
