#!/bin/bash

# Ensure temp directory exists
mkdir -p temp

# Parameters
DATE="2024-11-10"
OBJ="ackley"
EXEC="bo_benchmark.py"

DIM=5  # Fixed dimension value
ITER=300
SAMPLER="random"  # Fixed sampler
SEEDS=(0 1 2 3 4)  # Seed values

# Run experiments for each seed
for SEED in "${SEEDS[@]}"; do
    # Get current time in the format "YYYYMMDD_HHMM"
    TIMESTAMP=$(date +"%Y%m%d_%H%M")
    LOG_FILE="temp/${TIMESTAMP}_${OBJ}_${SAMPLER}_seed${SEED}.log"
    
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