#!/usr/bin/env bash
set -euo pipefail

# Full QTIP 1MAD pipeline for any Llama model:
# 1) Hessian generation
# 2) Quantization (1MAD)
# 3) Convert to HF format
# 4) Perplexity eval
# 5) Zeroshot eval
#
# Usage:
#   ./run_qtip_1mad_pipeline.sh <MODEL_ID> <RUN_TAG> [GPU_LIST]
#
# Example:
#   ./run_qtip_1mad_pipeline.sh meta-llama/Llama-3.1-8B l31_8b_2bit_1mad 0,1

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <MODEL_ID> <RUN_TAG> [GPU_LIST]"
  exit 1
fi

MODEL_ID="$1"
RUN_TAG="$2"
GPU_LIST="${3:-0,1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/home/user/Documents/qtip_experiments/.env-qtip/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Python env not found or not executable: $PY"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_LIST"

# Hessians are keyed by model ID so they can be reused across runs
HESS_MODEL_KEY="${MODEL_ID//\//_}"
HESS_DIR="$ROOT_DIR/hess/$HESS_MODEL_KEY"
CKPT_DIR="$ROOT_DIR/checkpoint/$RUN_TAG"
HF_DIR="$ROOT_DIR/hf/$RUN_TAG"
LOG_DIR="$ROOT_DIR/log"
LOG_FILE="$LOG_DIR/$RUN_TAG.log"

mkdir -p "$HESS_DIR" "$CKPT_DIR" "$HF_DIR" "$LOG_DIR"

echo "[INFO] Model: $MODEL_ID"
echo "[INFO] Run tag: $RUN_TAG"
echo "[INFO] GPUs: $CUDA_VISIBLE_DEVICES"
echo "[INFO] Log: $LOG_FILE"

# Optional dependency safety (no-op if already installed)
"$PY" -m pip install -q sentencepiece protobuf glog

{
  echo "===== STAGE 1: Hessian generation ====="
  if ls "$HESS_DIR"/*.pt 1>/dev/null 2>&1; then
    echo "[SKIP] Hessians for $MODEL_ID already exist in $HESS_DIR, skipping generation."
  else
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
      -m quantize_llama.input_hessian_llama \
      --base_model "$MODEL_ID" \
      --save_path "$HESS_DIR" \
      --devset_size 512 \
      --large_batch_size 256 \
      --batch_size 2 \
      --ctx_size 4096 \
      --sample_proc 8
  fi

  echo "===== STAGE 2: Quantization (1MAD) ====="
  "$PY" -m quantize_llama.quantize_finetune_llama \
    --save_path "$CKPT_DIR" \
    --codebook bitshift \
    --base_model "$MODEL_ID" \
    --in_hess_path "$HESS_DIR" \
    --scale_override 0.9 \
    --ft_epochs 0 \
    --td_x 16 \
    --td_y 16 \
    --L 16 \
    --K 2 \
    --V 1 \
    --decode_mode "1mad" \
    --tlut_bits 0

  echo "===== STAGE 3: Convert quantized model to HF ====="
  "$PY" -m quantize_llama.hfize_llama \
    --quantized_path "$CKPT_DIR" \
    --hf_output_path "$HF_DIR"

  echo "===== STAGE 4: Evaluate perplexity ====="
  "$PY" -m eval.eval_ppl \
    --hf_path "$HF_DIR"

  echo "===== STAGE 5: Evaluate zeroshot ====="
  "$PY" -m eval.eval_zeroshot \
    --tasks gsm8k \
    --batch_size 16 \
    --hf_path "$HF_DIR"

  echo "===== DONE ====="
} 2>&1 | tee -a "$LOG_FILE"
