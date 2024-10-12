#!/bin/bash

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/

# Params
ITER=$((500))  # Number of iterations for samplers
SEED_START=0  # Starting seed value
SEED_END=4  # Ending seed value (10 seeds in total)

# Overwrite config.ini file
config_file="config.ini"

# Loop through seeds for ParafacSampler
for SEED in $(seq $SEED_START $SEED_END); do

    # Run the Python script locally and log the output
    echo "Running experiment with seed $SEED..."
    python3 experiments/2024-10-12/bo_parafac.py \
        --seed $SEED \
        --iter_bo $ITER \
        --cp_rank 2 \
        --cp_als_iterations 100 \
        --cp_mask_ratio 0.1 \
        --cp_random_dist_type uniform \
        --acq_trade_off_param 2.0 \
        --acq_batch_size 1
done