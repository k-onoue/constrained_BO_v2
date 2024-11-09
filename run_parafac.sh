#!/bin/bash
# set -x  # Enable debug mode

DATE="2024-11-09"
OBJ="sphere"
EXEC="bo_parafac.py"

DIM=5
ITER=300
TRADE_OFF_PARAM=1
CP_MASK_RATIO=0.2
DECOMP_NUM=10

for seed in {0..4}; do
    nohup python3 experiments/$DATE/$OBJ/$EXEC \
            --dimensions $DIM \
            --iter_bo $ITER \
            --acq_trade_off_param $TRADE_OFF_PARAM \
            --cp_mask_ratio $CP_MASK_RATIO \
            --decomp_num $DECOMP_NUM \
            --seed $seed > output_$seed.log 2>&1 &
done