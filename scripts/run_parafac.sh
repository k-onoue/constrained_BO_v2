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

# Which date to use for the experiments
DATE="2024-10-17"

# Sampler specific parameters
CP_RANK=2  # CP rank for the Parafac decomposition
CP_ALS_ITERATIONS=100  # Number of ALS iterations for Parafac decomposition
CP_MASK_RATIO=0.33  # Mask ratio for Parafac sampling
CP_RANDOM_DIST_TYPE="uniform"  # Distribution type for random sampling in Parafac
ACQ_TRADE_OFF_PARAM=2.0  # Acquisition function trade-off parameter
ACQ_BATCH_SIZE=1  # Batch size for acquisition function optimization

# Define the list of maps
MAPS=(2 3)  # You can define these as specific arrays in your Python script

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

# Loop through maps and seeds for ParafacSampler
for MAP in "${MAPS[@]}"; do
    for SEED in $(seq $SEED_START $SEED_END); do
        # Set up experiment name and log file paths
        EXPERIMENT_NAME="parafac_benchmark_${MAP}_seed${SEED}"

        # Run each experiment in parallel using sbatch
        sbatch --job-name="${EXPERIMENT_NAME}" \
               --output="${TEMP}/${EXPERIMENT_NAME}_%j.log" \
               --cpus-per-task=$CPUS_PER_TASK \
               --partition=$PARTITION \
               --time=$TIME \
               --wrap="python3 experiments/$DATE/bo_parafac.py --map $MAP --seed $SEED --iter_bo $ITER --cp_rank $CP_RANK --cp_als_iterations $CP_ALS_ITERATIONS --cp_mask_ratio $CP_MASK_RATIO --cp_random_dist_type $CP_RANDOM_DIST_TYPE --acq_trade_off_param $ACQ_TRADE_OFF_PARAM --acq_batch_size $ACQ_BATCH_SIZE"
    done
done