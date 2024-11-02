#!/bin/bash

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/
mkdir -p temp/

DATE="2024-11-01"

# Params
ITER=500  # Number of iterations for samplers
# ITER=3000  # Number of iterations for samplers
SEED_START=0  # Starting seed value
SEED_END=4  # Ending seed value (5 seeds in total)
TEMP="temp"  # Temporary directory for log files

# Sampler list excluding Bruteforce
SAMPLERS=("random" "tpe" "gp")
# SAMPLERS=("random")

# Bruteforce will be run separately
BRUTEFORCE_SAMPLER="bruteforce"

# Define the list of dimensions
# DIMENSIONS=(2 3 5)  # You can modify this to fit your needs
DIMENSIONS=(2 3 5 7)

# Loop through dimensions, samplers, and seeds
for DIM in "${DIMENSIONS[@]}"; do
    for SAMPLER in "${SAMPLERS[@]}"; do
        for SEED in $(seq $SEED_START $SEED_END); do

            # Set up experiment name and log file paths
            EXPERIMENT_NAME="benchmark_${SAMPLER}_dim${DIM}_seed${SEED}"
            LOG_FILE="${TEMP}/${EXPERIMENT_NAME}.log"

            echo "Running experiment with sampler $SAMPLER, dimension $DIM, seed $SEED..."

            # Run each experiment locally and log the output
            python3 experiments/${DATE}/sphere/bo_benchmark.py \
                --sampler $SAMPLER \
                --dimensions $DIM \
                --seed $SEED \
                --iter_bo $ITER \
                > "$LOG_FILE" 2>&1

            echo "Log saved to $LOG_FILE"
        done
    done
done

# # Run Bruteforce separately for each dimension (no seed loop)
# for DIM in "${DIMENSIONS[@]}"; do
#     EXPERIMENT_NAME="benchmark_bruteforce_dim${DIM}"
#     LOG_FILE="${TEMP}/${EXPERIMENT_NAME}.log"

#     echo "Running experiment with Bruteforce on dimension $DIM..."

#     # Run the Bruteforce experiment locally and log the output
#     python3 experiments/2024-10-25/ackley/bo_benchmark.py \
#         --sampler $BRUTEFORCE_SAMPLER \
#         --dimensions $DIM \
#         --iter_bo $ITER \
#         > "$LOG_FILE" 2>&1

#     echo "Log saved to $LOG_FILE"
# done
