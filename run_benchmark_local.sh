#!/bin/bash

# SLURM Resource configuration
CPUS_PER_TASK=4  # Number of CPUs per task
PARTITION="cluster_short"  # Partition name
TIME="4:00:00"  # Maximum execution time

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
MAPS=("map1" "map2" "map3")  # You can define these as specific arrays in your Python script

# Overwrite config.ini file
config_file="config.ini"

config_content="[paths]
project_dir = /work/keisuke-o/ws/constrained_BO_v2
data_dir = %(project_dir)s/data
results_dir = %(project_dir)s/results
logs_dir = %(results_dir)s/logs
dbs_dir = %(results_dir)s/dbs"

# Overwrite config.ini file if necessary
echo "$config_content" > $config_file

# Confirm the overwrite
echo "config.ini has been overwritten with the following content:"
cat $config_file

# Loop through maps, samplers, and seeds
for MAP in "${MAPS[@]}"; do
    for SAMPLER in "${SAMPLERS[@]}"; do
        for SEED in $(seq $SEED_START $SEED_END); do

            # Set up experiment name and log file paths
            EXPERIMENT_NAME="benchmark_${SAMPLER}_${MAP}_seed${SEED}"

            # Run each experiment in parallel using sbatch
            sbatch --job-name="${EXPERIMENT_NAME}" \
                   --output="${TEMP}/${EXPERIMENT_NAME}_%j.log" \
                   --cpus-per-task=$CPUS_PER_TASK \
                   --partition=$PARTITION \
                   --time=$TIME \
                   --wrap="python3 experiments/2024-10-13/bo_benchmark.py --sampler $SAMPLER --map $MAP --seed $SEED --iter_bo $ITER"
        done
    done
done

# Run Bruteforce separately for each map (no seed loop)
for MAP in "${MAPS[@]}"; do
    EXPERIMENT_NAME="benchmark_bruteforce_${MAP}"

    # Run the Bruteforce experiment using sbatch
    sbatch --job-name="${EXPERIMENT_NAME}" \
           --output="${TEMP}/${EXPERIMENT_NAME}_%j.log" \
           --cpus-per-task=$CPUS_PER_TASK \
           --partition=$PARTITION \
           --time=$TIME \
           --wrap="python3 experiments/2024-10-13/bo_benchmark.py --sampler $BRUTEFORCE_SAMPLER --map $MAP --iter_bo $ITER"
done
