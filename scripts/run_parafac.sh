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

# Define the list of maps
# MAPS=("map1" "map2" "map3")  # You can define these as specific arrays in your Python script
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
               --wrap="python3 experiments/2024-10-13/bo_parafac.py --map $MAP --seed $SEED --iter_bo $ITER --cp_rank 2 --cp_als_iterations 100 --cp_mask_ratio 0.1 --cp_random_dist_type uniform --acq_trade_off_param 2.0 --acq_batch_size 1"
    done
done
