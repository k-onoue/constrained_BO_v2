#!/bin/bash

# SLURM Resource configuration
CPUS_PER_TASK=26  # Maximum number of CPUs available per task on cluster nodes
PARTITION="cluster_short"  # Partition name
TIME="4:00:00"  # Maximum execution time

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p logs/

# Params
ITER=50
EXPERIMENTAL_ID="DIM_EXP"

# Overwrite config.ini file
config_file="config.ini"

config_content="[paths]
project_dir = /work/keisuke-o/ws/constrained_BO_v2
data_dir = %(project_dir)s/data
results_dir = %(project_dir)s/results
logs_dir = %(project_dir)s/logs/"

# Overwrite config.ini file only if necessary
echo "$config_content" > $config_file

# Confirm the overwrite
echo "config.ini has been overwritten with the following content:"
cat $config_file

# Parallelize over dims from 5 to 15
for DIM in {5..15}; do
    # Set up experiment name and log file paths
    EXPERIMENT_NAME="benchmark_cp_decomp_dim${DIM}"
    LOG_DIR="logs/${EXPERIMENTAL_ID}"

    # Run each experiment in parallel using sbatch
    sbatch --job-name="${EXPERIMENT_NAME}" \
           --output="${LOG_DIR}/${EXPERIMENT_NAME}_%j.log" \
           --cpus-per-task=$CPUS_PER_TASK \
           --partition=$PARTITION \
           --time=$TIME \
           --wrap="python3 experiments/2024-09-29/benchmark_cp_decomp.py --dim $DIM"
done