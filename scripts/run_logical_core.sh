#!/bin/bash

# SLURM Resource configuration
PARTITION="gpu_short"  # Partition name
TIME="4:00:00"  # Maximum execution time

# Params
LOGICAL_CORES_LIST=(4 8 12 16 20 24)  # List of logical cores to test
# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/
mkdir -p temp/

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

# Loop through each logical core setting
for LOGICAL_CORES in "${LOGICAL_CORES_LIST[@]}"; do
    # Set up experiment name and log file paths
    EXPERIMENT_NAME="parafac_benchmark_cores${LOGICAL_CORES}"

    # Run each experiment in parallel using sbatch
    sbatch --job-name="${EXPERIMENT_NAME}" \
           --output="${TEMP}/${EXPERIMENT_NAME}_%j.log" \
           --cpus-per-task=$LOGICAL_CORES \
           --partition=$PARTITION \
           --time=$TIME \
           --wrap="python3 experiments/2024-10-12/logical_core_benchmark.py --logical_cores $LOGICAL_CORES"
done
