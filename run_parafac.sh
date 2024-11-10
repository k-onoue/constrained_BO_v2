#!/bin/bash
# set -x  # Enable debug mode

DATE="2024-11-10"
OBJ="ackley"
EXEC="bo_parafac.py"

DIM=3
ITER=300
TRADE_OFF_PARAM=1
CP_MASK_RATIO=0.1
DECOMP_NUM=50
SEEDS=1  # 実行するシードの数

# Ensure temp directory exists
mkdir -p temp

# seed の設定
for seed in $(seq 0 $((SEEDS - 1))); do
    # Get current time in the format "YYYYMMDD_HHMM"
    TIMESTAMP=$(date +"%Y%m%d_%H%M")
    
    nohup python3 experiments/$DATE/$OBJ/$EXEC \
            --dimensions $DIM \
            --iter_bo $ITER \
            --acq_trade_off_param $TRADE_OFF_PARAM \
            --cp_mask_ratio $CP_MASK_RATIO \
            --decomp_num $DECOMP_NUM \
            --seed $seed > temp/${TIMESTAMP}_${OBJ}_dn${DECOMP_NUM}_$seed.log 2>&1 &
done