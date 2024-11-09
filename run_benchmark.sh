#!/bin/bash

# Ensure temp directory exists
mkdir -p temp

# Parameters
DATE="2024-11-09"
OBJ="sphere"
EXEC="bo_benchmark.py"

DIM=5  # Fixed dimension value
ITER=300
SAMPLER="random"  # Fixed sampler
SEEDS=(0 1 2 3 4)  # Seed values

# Run experiments for each seed
for SEED in "${SEEDS[@]}"; do
    LOG_FILE="temp/benchmark_${SAMPLER}_dim${DIM}_seed${SEED}.log"
    echo "Running experiment with sampler $SAMPLER, dimension $DIM, seed $SEED..."
    
    # Run each experiment in the background
    nohup python3 experiments/$DATE/$OBJ/$EXEC \
        --sampler $SAMPLER \
        --dimensions $DIM \
        --seed $SEED \
        --iter_bo $ITER \
        > "$LOG_FILE" 2>&1 &
    
    echo "Log saved to $LOG_FILE"
done