#!/bin/bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
##### Ori
EXP_NAME="${EXP_NAME:-sap_G4_aug}"
TRAIN_ROOT="${TRAIN_ROOT:-Dataset/train}"
VAL_ROOT="${VAL_ROOT:-Dataset/val}"
SEED="${SEED:-3407}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"

TRAIN_CLASSES_STR="${TRAIN_CLASSES:-progan stable_diffusion_v_1_4 biggan adm}"
VAL_CLASSES_STR="${VAL_CLASSES:-progan stable_diffusion_v_1_4 biggan adm}"

read -r -a TRAIN_CLASSES_ARR <<< "${TRAIN_CLASSES_STR}"
read -r -a VAL_CLASSES_ARR <<< "${VAL_CLASSES_STR}"

echo "[SICA] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[SICA] experiment=${EXP_NAME}"
echo "[SICA] train_root=${TRAIN_ROOT} val_root=${VAL_ROOT}"
echo "[SICA] train_classes=${TRAIN_CLASSES_STR}"
echo "[SICA] val_classes=${VAL_CLASSES_STR}"

python train.py \
    --experiment_name "${EXP_NAME}" \
    --train_data_root "${TRAIN_ROOT}" \
    --val_data_root "${VAL_ROOT}" \
    --train_classes "${TRAIN_CLASSES_ARR[@]}" \
    --val_classes "${VAL_CLASSES_ARR[@]}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --use_data_augument \
    --rank 8 \
    --target_layer_idx 18
    "$@"