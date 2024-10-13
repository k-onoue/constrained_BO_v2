#!/bin/bash

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/
mkdir -p temp/

# Params
ITER=500  # Number of iterations for samplers
SEED_START=0  # Starting seed value
SEED_END=4  # Ending seed value (5 seeds in total)
TEMP="temp"  # Temporary directory for log files

# Sampler list excluding Bruteforce
SAMPLERS=("random" "tpe" "gp")

# Bruteforce will be run separately
BRUTEFORCE_SAMPLER="bruteforce"

# Define the list of maps
MAPS=(2)  # You can define these as specific arrays in your Python script

# Loop through maps, samplers, and seeds
for MAP in "${MAPS[@]}"; do
    for SAMPLER in "${SAMPLERS[@]}"; do
        for SEED in $(seq $SEED_START $SEED_END); do

            # Set up experiment name and log file paths
            EXPERIMENT_NAME="benchmark_${SAMPLER}_${MAP}_seed${SEED}"
            LOG_FILE="${TEMP}/${EXPERIMENT_NAME}.log"

            echo "Running experiment with sampler $SAMPLER, map $MAP, seed $SEED..."

            # Run each experiment locally and log the output
            python3 experiments/2024-10-13/bo_benchmark.py \
                --sampler $SAMPLER \
                --map $MAP \
                --seed $SEED \
                --iter_bo $ITER \
                > "$LOG_FILE" 2>&1

            echo "Log saved to $LOG_FILE"
        done
    done
done

# Run Bruteforce separately for each map (no seed loop)
for MAP in "${MAPS[@]}"; do
    EXPERIMENT_NAME="benchmark_bruteforce_${MAP}"
    LOG_FILE="${TEMP}/${EXPERIMENT_NAME}.log"

    echo "Running experiment with Bruteforce on map $MAP..."

    # Run the Bruteforce experiment locally and log the output
    python3 experiments/2024-10-13/bo_benchmark.py \
        --sampler $BRUTEFORCE_SAMPLER \
        --map $MAP \
        --iter_bo $ITER \
        > "$LOG_FILE" 2>&1

    echo "Log saved to $LOG_FILE"
done
