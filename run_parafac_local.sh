#!/bin/bash

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/
mkdir -p temp/

# Parameters
ITER=500  # Number of iterations for ParafacSampler
SEED_START=0  # Starting seed value
SEED_END=4  # Ending seed value (5 seeds in total)
TEMP="temp"  # Temporary directory for log files

# Define the list of dimensions, cp_rank, cp_mask_ratio, trade_off_param values, and distribution type
DIMENSIONS=(2 3 5 7)
CP_RANKS=(1 2 3 4)
CP_MASK_RATIOS=(0.33)
TRADE_OFF_PARAMS=(3)
CP_RANDOM_DIST_TYPE="normal"  # Distribution type for random sampling

# Loop through dimensions, cp_rank, cp_mask_ratio, trade_off_param, and seeds
for DIM in "${DIMENSIONS[@]}"; do
    for CP_RANK in "${CP_RANKS[@]}"; do
        for CP_MASK_RATIO in "${CP_MASK_RATIOS[@]}"; do
            for TRADE_OFF_PARAM in "${TRADE_OFF_PARAMS[@]}"; do
                for SEED in $(seq $SEED_START $SEED_END); do

                    # Set up experiment name and log file paths
                    EXPERIMENT_NAME="benchmark_parafac_dim${DIM}_rank${CP_RANK}_mask${CP_MASK_RATIO}_tradeoff${TRADE_OFF_PARAM}_seed${SEED}"
                    LOG_FILE="${TEMP}/${EXPERIMENT_NAME}.log"

                    echo "Running experiment with ParafacSampler, dimension $DIM, cp_rank $CP_RANK, mask_ratio $CP_MASK_RATIO, trade_off_param $TRADE_OFF_PARAM, seed $SEED..."

                    # Run each experiment and log the output
                    python3 experiments/2024-10-25/ackley/bo_parafac.py \
                        --dimensions $DIM \
                        --cp_rank $CP_RANK \
                        --cp_mask_ratio $CP_MASK_RATIO \
                        --acq_trade_off_param $TRADE_OFF_PARAM \
                        --seed $SEED \
                        --iter_bo $ITER \
                        --cp_random_dist_type $CP_RANDOM_DIST_TYPE \
                        > "$LOG_FILE" 2>&1

                    echo "Log saved to $LOG_FILE"
                done
            done
        done
    done
done
