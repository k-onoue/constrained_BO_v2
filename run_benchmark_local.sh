#!/bin/bash

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/

# Params
ITER=500  # Number of iterations for samplers
SEED_START=0  # Starting seed value
SEED_END=4  # Ending seed value (10 seeds in total)

# Sampler list excluding Bruteforce
SAMPLERS=("random" "tpe")

# Bruteforce will be run separately
BRUTEFORCE_SAMPLER="bruteforce"

# Overwrite config.ini file
config_file="config.ini"

# Loop through samplers and seeds, excluding Bruteforce from seeding loop
for SAMPLER in "${SAMPLERS[@]}"; do
    for SEED in $(seq $SEED_START $SEED_END); do

        # Run each experiment locally and log the output
        echo "Running experiment with sampler $SAMPLER and seed $SEED..."
        python3 experiments/2024-10-12/bo_benchmark.py \
            --sampler $SAMPLER \
            --seed $SEED \
            --iter_bo $ITER
    done
done

# Run the Bruteforce experiment locally and log the output
echo "Running experiment with sampler $BRUTEFORCE_SAMPLER..."
python3 experiments/2024-10-12/bo_benchmark.py \
    --sampler $BRUTEFORCE_SAMPLER \