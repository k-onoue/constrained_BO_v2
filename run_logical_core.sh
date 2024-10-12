#!/bin/bash

# SLURM Resource configuration
CPUS_PER_TASK=4  # Number of CPUs per task
PARTITION="cluster_long"  # Partition name
TIME="10:00:00"  # Maximum execution time

# Create temp directory if it doesn't exist
mkdir -p temp/

# Params
LOGICAL_CORES_LIST=(4 8 12 16 20 24)  # List of logical cores to test
TEMP="temp"  # Temporary directory for log files

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
