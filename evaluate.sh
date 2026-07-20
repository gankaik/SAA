#!/bin/bash
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
EXP_NAME="sap_G4_aug"
EVAL_DATA_BASE="**EVAL_DATA_PATH**"
MODEL_BASE="**MODEL_PATH**"



eval_data_root_list=(
    "${EVAL_DATA_BASE}/eval_UFD"    
    "${EVAL_DATA_BASE}/eval_GenImage"
    "${EVAL_DATA_BASE}/eval_AIGI"
)
WEIGHT_PATH="${MODEL_BASE}/train_sap/model/sap_best_model.pth"

for eval_data_root in "${eval_data_root_list[@]}"; do
    echo "Evaluating on ${eval_data_root}..."
    python evaluate.py \
        --experiment_name "${EXP_NAME}/train_sap" \
        --eval_data_root ${eval_data_root} \
        --seed 3407 \
        --weights "${WEIGHT_PATH}" \
        --batch_size 16 \
        --rank 8
done
#-----------------------