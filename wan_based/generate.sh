#!/bin/bash
# ==============================================================================
# MIGA generation with Wan2.1-1.3B
# Usage: bash generate.sh
# ==============================================================================

# ========== Modify the following paths ==========
CKPT_DIR="/path/to/Wan2.1-T2V-1.3B"
SAVE_DIR="./outputs"
# ================================================

python generate.py \
    --ckpt_dir ${CKPT_DIR} \
    --miga_config ../configs/wan2.1_1.3B.yaml \
    --prompt "A fluffy Corgi dog trots happily across a lush green lawn." \
    --save_dir ${SAVE_DIR} \
    --exp_name demo
