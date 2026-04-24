#!/usr/bin/env bash
set -euo pipefail
# Full QTIP 1MAD pipeline for Llama/Qwen3 models:
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
#   ./run_qtip_1mad_pipeline.sh meta-llama/Llama-3.1-8B l31_8b_2bit_1mad 0,1,2,3,4,5,6,7,8
#   ./run_qtip_1mad_pipeline.sh Qwen/Qwen3-8B qwen3_8b_2bit_1mad 0,1,2,3

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <MODEL_ID> <RUN_TAG> [GPU_LIST] [TD_X] [TD_Y] [DECODE_MODE] [SKIP_LAYERS]"
  exit 1
fi

MODEL_ID="$1"
RUN_TAG="$2"

# Auto-detect all available GPUs if not specified
DEFAULT_GPUS=$(nvidia-smi --list-gpus | awk '{print NR-1}' | paste -sd,)
GPU_LIST="${3:-$DEFAULT_GPUS}"
TD_X="${4:-16}"
TD_Y="${5:-16}"
DECODE_MODE="${6:-1mad}"
SKIP_LAYERS="${7:-}"   # whole layer indices: "0,1,31"
SKIP_SUBLAYERS="${8:-}" # specific sublayers: "5_v,10_down"

# Count GPUs correctly by splitting on comma
IFS=',' read -ra GPU_ARRAY <<< "$GPU_LIST"
NUM_GPUS=${#GPU_ARRAY[@]}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect Python from active venv or fall back to known path
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PY="$VIRTUAL_ENV/bin/python"
else
  PY="/home/anazir/qtip_experiments/.env-qtip/bin/python"
fi

if [[ ! -x "$PY" ]]; then
  echo "Python env not found or not executable: $PY"
  echo "Please activate your venv: source /home/anazir/qtip_experiments/.env-qtip/bin/activate"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_LIST"

# Data dirs live on /local_ssd2 for space; logs stay local
DATA_ROOT="/local_ssd2/anazir/qtip"
HESS_MODEL_KEY="${MODEL_ID//\//_}"
HESS_DIR="$DATA_ROOT/hess/$HESS_MODEL_KEY"
CKPT_DIR="$DATA_ROOT/checkpoint/$RUN_TAG"
HF_DIR="$DATA_ROOT/hf/$RUN_TAG"
LOG_DIR="$ROOT_DIR/log"
LOG_FILE="$LOG_DIR/$RUN_TAG.log"
mkdir -p "$HESS_DIR" "$CKPT_DIR" "$HF_DIR" "$LOG_DIR"

echo "[INFO] Model:    $MODEL_ID"
echo "[INFO] Run tag:  $RUN_TAG"
echo "[INFO] GPUs:     $CUDA_VISIBLE_DEVICES"
echo "[INFO] Num GPUs: $NUM_GPUS"
echo "[INFO] Python:   $PY"
echo "[INFO] Log:      $LOG_FILE"
echo "[INFO] TD_X:     $TD_X"
echo "[INFO] TD_Y:     $TD_Y"
echo "[INFO] Decode:   $DECODE_MODE"
echo "[INFO] Skip layers:     ${SKIP_LAYERS:-none}"
echo "[INFO] Skip sublayers:  ${SKIP_SUBLAYERS:-none}"

# Optional dependency safety (no-op if already installed)
"$PY" -m pip install -q sentencepiece protobuf glog

{
  echo "===== STAGE 1: Hessian generation ====="
  if ls "$HESS_DIR"/*.pt 1>/dev/null 2>&1; then
    echo "[SKIP] Hessians for $MODEL_ID already exist in $HESS_DIR, skipping generation."
  else
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=$NUM_GPUS \
      -m quantize_llama.input_hessian_llama \
      --base_model "$MODEL_ID" \
      --save_path "$HESS_DIR" \
      --devset_size 512 \
      --large_batch_size 256 \
      --batch_size 32 \
      --ctx_size 4096 \
      --sample_proc 8
  fi

  echo "===== STAGE 2: Quantization ====="
  "$PY" -m quantize_llama.quantize_finetune_llama \
    --save_path "$CKPT_DIR" \
    --codebook bitshift \
    --base_model "$MODEL_ID" \
    --in_hess_path "$HESS_DIR" \
    --scale_override 0.9 \
    --ft_epochs 0 \
    --td_x $TD_X \
    --td_y $TD_Y \
    --L 16 \
    --K 2 \
    --V 1 \
    --decode_mode "$DECODE_MODE" \
    --tlut_bits 0 \
    ${SKIP_LAYERS:+--skip_layers "$SKIP_LAYERS"} \
    ${SKIP_SUBLAYERS:+--skip_list "$SKIP_SUBLAYERS"}

  echo "===== STAGE 3: Convert quantized model to HF ====="
  "$PY" -m quantize_llama.hfize_llama \
    --quantized_path "$CKPT_DIR" \
    --hf_output_path "$HF_DIR"

  echo "===== Model Size Report ====="
  "$PY" -c "
import os, sys, torch, json
from safetensors.torch import load_file

hf_dir = '$HF_DIR'
base_model = '$MODEL_ID'

# Total safetensors file size on disk
total_disk = sum(
    os.path.getsize(os.path.join(hf_dir, f))
    for f in os.listdir(hf_dir) if f.endswith('.safetensors')
)

# Break down by parameter type
st = load_file(os.path.join(hf_dir, 'model.safetensors'))
quant_bytes = 0
non_quant_bytes = 0
non_quant_params = 0
total_params = 0

for name, tensor in st.items():
    nbytes = tensor.nelement() * tensor.element_size()
    total_params += tensor.nelement()
    if 'trellis' in name:
        quant_bytes += nbytes
    else:
        non_quant_bytes += nbytes
        non_quant_params += tensor.nelement()

# FP16 model size for comparison
from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained(base_model)
    # Rough param count from config
    fp16_size_gb = None
except:
    fp16_size_gb = None

print(f'  Quantized model disk size:  {total_disk / (1 << 30):.2f} GB')
print(f'  Quantized weights (trellis): {quant_bytes / (1 << 20):.1f} MB')
print(f'  Non-quantized (embed/norm/lm_head): {non_quant_bytes / (1 << 20):.1f} MB')
print(f'  Total parameters: {total_params:,}')
print(f'  Effective bits/param (overall): {total_disk * 8 / total_params:.2f}')
"

  # echo "===== STAGE 4: Evaluate perplexity ====="
  # accelerate launch --multi_gpu --num_processes $NUM_GPUS -m eval.eval_ppl \
  #   --hf_path "$HF_DIR" \
  #   --manifest

  echo "===== STAGE 5: Evaluate zeroshot ====="
  accelerate launch --multi_gpu --num_processes $NUM_GPUS -m eval.eval_zeroshot \
    --tasks gsm8k,lambada_openai \
    --batch_size 16 \
    --manifest_model \
    --hf_path "$HF_DIR"

  echo "===== DONE ====="
} 2>&1 | tee -a "$LOG_FILE"
