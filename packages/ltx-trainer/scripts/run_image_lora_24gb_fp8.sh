#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Edit these paths
# ---------------------------------------------------------------------------
MODEL_PATH="/home/pyro/models/comfy/checkpoints/ltx-2.3-22b-dev.safetensors"
GEMMA_DIR="/home/pyro/models/gemma/gemma-3-12b-it-qat-q4_0-unquantized"

# Option A: point directly to an existing JSON/JSONL/CSV metadata file.
# Note: the trainer expects media paths to live under the metadata file's parent directory.
DATASET_METADATA=""

# Option B: point to a folder containing files like image.png + image.txt.
# If this is set, the script will build JSONL for you and use that automatically.
PAIR_DATASET_ROOT="/home/pyro/datasets/iterator/cont400/headsit/2legs_on_floor"

PREPROCESSED_ROOT="/home/pyro/repos/LTX-2/_data"
OUTPUT_DIR="/home/pyro/repos/LTX-2/_data/_out"

# Start conservative on 24GB. Change to "960x544x1" or similar if needed.
RESOLUTION_BUCKETS="512x512x1"

# Training knobs
LORA_RANK="4"
LORA_ALPHA="4"
TRAIN_STEPS="2000"
GRAD_ACCUM_STEPS="4"
LEARNING_RATE="1e-4"
CHECKPOINT_INTERVAL="50"
NUM_DATALOADER_WORKERS="2"

# Important: the trainer casts LoRA runs to bf16 before optional trainer-side quantization.
# That means an FP8 checkpoint alone does not keep memory low here.
# For 24GB, start with int4-quanto. If it still OOMs, last-resort options are int2-quanto
# or a smaller base checkpoint.
QUANTIZATION="int8-quanto"

# Mode: preprocess | train | all
MODE="${1:-all}"

# ---------------------------------------------------------------------------
# Paths derived from the values above
# ---------------------------------------------------------------------------
GENERATED_DIR="${TRAINER_DIR}/configs/generated"
GENERATED_CONFIG="${GENERATED_DIR}/ltx2_image_lora_24gb_fp8.local.yaml"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "missing file: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "missing directory: $path"
}

case "${MODE}" in
  preprocess|train|all) ;;
  *) die "mode must be one of: preprocess, train, all" ;;
esac

require_file "${MODEL_PATH}"
require_dir "${GEMMA_DIR}"

mkdir -p "${GENERATED_DIR}" "${PREPROCESSED_ROOT}" "${OUTPUT_DIR}"

DATASET_PATH="${DATASET_METADATA}"
if [[ -n "${PAIR_DATASET_ROOT}" && "${PAIR_DATASET_ROOT}" != "/ABS/PATH/TO/IMAGE_TXT_DATASET" ]]; then
  require_dir "${PAIR_DATASET_ROOT}"
  AUTO_DATASET_JSONL="${PAIR_DATASET_ROOT}/.ltx_dataset.jsonl"
  (
    cd "${TRAINER_DIR}"
    uv run python scripts/build_image_jsonl_from_txt_pairs.py \
      "${PAIR_DATASET_ROOT}" \
      --output "${AUTO_DATASET_JSONL}"
  )
  DATASET_PATH="${AUTO_DATASET_JSONL}"
fi

[[ -n "${DATASET_PATH}" ]] || die "set DATASET_METADATA or PAIR_DATASET_ROOT near the top of this script"
require_file "${DATASET_PATH}"

cat > "${GENERATED_CONFIG}" <<EOF
model:
  model_path: "${MODEL_PATH}"
  text_encoder_path: "${GEMMA_DIR}"
  training_mode: "lora"
  load_checkpoint: null

lora:
  rank: ${LORA_RANK}
  alpha: ${LORA_ALPHA}
  dropout: 0.0
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 1
  with_audio: false
  audio_latents_dir: "audio_latents"

optimization:
  learning_rate: ${LEARNING_RATE}
  steps: ${TRAIN_STEPS}
  batch_size: 1
  gradient_accumulation_steps: ${GRAD_ACCUM_STEPS}
  max_grad_norm: 1.0
  optimizer_type: "adamw8bit"
  scheduler_type: "linear"
  scheduler_params: {}
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: "${QUANTIZATION}"
  load_text_encoder_in_8bit: true
  blocks_to_swap: 12
  use_pinned_memory_for_block_swap: false

data:
  preprocessed_data_root: "${PREPROCESSED_ROOT}"
  num_dataloader_workers: ${NUM_DATALOADER_WORKERS}

validation:
  prompts: ["a pretty and skinny woman doing a headsit pose at home in her room, wearing casual gothic clothing. amateur candid shot", "a very skinny korean actress doing a headsit pose, wearing sleek expensive fashion. amateur candid shot. at the mall.", "a extremely skinny 20-year-old female topmodel doing a headsit pose, mixed Asian-European descent, amateur photo, low-lit, overexposure, Low-resolution photo, shot on a mobile phone, on a yoga mat inside a yoga studio"]
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null
  video_dims: [544, 960, 1]
  frame_rate: 1
  seed: 42
  inference_steps: 30
  interval: null
  videos_per_prompt: 1
  guidance_scale: 4.0
  stg_scale: 1.0
  stg_blocks: [29]
  stg_mode: "stg_v"
  generate_audio: false
  skip_initial_validation: true

checkpoints:
  interval: ${CHECKPOINT_INTERVAL}
  keep_last_n: -1
  precision: "bfloat16"

flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"
  timestep_sampling_params: {}

hub:
  push_to_hub: false
  hub_model_id: null

wandb:
  enabled: false
  project: "ltx-2-trainer"
  entity: null
  tags: ["ltx2.3", "image-only", "lora", "fp8-base"]
  log_validation_videos: false

seed: 42
output_dir: "${OUTPUT_DIR}"
EOF

if [[ "${MODE}" == "preprocess" || "${MODE}" == "all" ]]; then
  (
    cd "${TRAINER_DIR}"
    uv run python scripts/process_dataset.py \
      "${DATASET_PATH}" \
      --resolution-buckets "${RESOLUTION_BUCKETS}" \
      --model-path "${MODEL_PATH}" \
      --text-encoder-path "${GEMMA_DIR}" \
      --batch-size 1 \
      --output-dir "${PREPROCESSED_ROOT}" \
      --load-text-encoder-in-8bit
  )
fi

if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
  (
    cd "${TRAINER_DIR}"
    uv run python scripts/train.py "${GENERATED_CONFIG}"
  )
fi

printf 'Config written to %s\n' "${GENERATED_CONFIG}"
