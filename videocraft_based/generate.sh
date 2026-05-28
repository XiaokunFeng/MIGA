#!/bin/bash
# ==============================================================================
# MIGA generation with VideoCrafter2
# Usage: bash generate.sh
# ==============================================================================

# ========== Modify the following paths ==========
CKPT_PATH="/path/to/videocrafter_models/base_512_v2/model.ckpt"
SAVE_DIR="./outputs"
# ================================================

python generate.py \
    --ckpt_path ${CKPT_PATH} \
    --miga_config ../configs/videocrafter2.yaml \
    --prompt "An astronaut floating in space, high quality, 4K resolution." \
    --save_dir ${SAVE_DIR} \
    --exp_name demo
