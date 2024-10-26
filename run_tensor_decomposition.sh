#!/bin/bash
set -x  # Enable debug mode

# Params
LOGICAL_CORES=8  # Assign 8 logical cores to each process
DATE="2024-10-25"  # Experiment date as a variable
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/
mkdir -p temp/

# Rank list
RANK_LIST=(0 1 2)

# Tensor Ring
EXPERIMENT_NAME="tensor_ring_benchmark"
for RANK in "${RANK_LIST[@]}"; do
    LOG_FILE="temp/${EXPERIMENT_NAME}_rank${RANK}.log"
    echo "Running Tensor Ring Benchmark with logical cores $LOGICAL_CORES, rank $RANK..."
    taskset -c 0-7 python3 experiments/${DATE}/tensor_decomposition/tensor_ring_benchmark.py --logical_cores $LOGICAL_CORES --rank $RANK > "$LOG_FILE" 2>&1 &
    echo "Log saved to $LOG_FILE"
done

# Tensor Ring ALS Sampled
EXPERIMENT_NAME="tensor_ring_als_sampled_benchmark"
for RANK in "${RANK_LIST[@]}"; do
    LOG_FILE="temp/${EXPERIMENT_NAME}_rank${RANK}.log"
    echo "Running Tensor Ring ALS Sampled Benchmark with logical cores $LOGICAL_CORES, rank $RANK..."
    taskset -c 8-15 python3 experiments/${DATE}/tensor_decomposition/tensor_ring_als_sampled_benchmark.py --logical_cores $LOGICAL_CORES --rank $RANK > "$LOG_FILE" 2>&1 &
    echo "Log saved to $LOG_FILE"
done

# Tensor Ring ALS
EXPERIMENT_NAME="tensor_ring_als_benchmark"
for RANK in "${RANK_LIST[@]}"; do
    LOG_FILE="temp/${EXPERIMENT_NAME}_rank${RANK}.log"
    echo "Running Tensor Ring ALS Benchmark with logical cores $LOGICAL_CORES, rank $RANK..."
    taskset -c 16-23 python3 experiments/${DATE}/tensor_decomposition/tensor_ring_als_benchmark.py --logical_cores $LOGICAL_CORES --rank $RANK > "$LOG_FILE" 2>&1 &
    echo "Log saved to $LOG_FILE"
done

# # Tensor Train
# EXPERIMENT_NAME="tensor_train_benchmark"
# for RANK in "${RANK_LIST[@]}"; do
#     LOG_FILE="temp/${EXPERIMENT_NAME}_rank${RANK}.log"
#     echo "Running Tensor Train Benchmark with logical cores $LOGICAL_CORES, rank $RANK..."
#     taskset -c 24-31 python3 experiments/${DATE}/tensor_decomposition/tensor_train_benchmark.py --logical_cores $LOGICAL_CORES --rank $RANK > "$LOG_FILE" 2>&1 &
#     echo "Log saved to $LOG_FILE"
# done

# # Parafac
# EXPERIMENT_NAME="parafac_benchmark"
# for RANK in "${RANK_LIST[@]}"; do
#     LOG_FILE="temp/${EXPERIMENT_NAME}_rank${RANK}.log"
#     echo "Running Parafac Benchmark with logical cores $LOGICAL_CORES, rank $RANK..."
#     taskset -c 32-39 python3 experiments/${DATE}/tensor_decomposition/parafac_benchmark.py --logical_cores $LOGICAL_CORES --rank $RANK > "$LOG_FILE" 2>&1 &
#     echo "Log saved to $LOG_FILE"
# done

# Wait for all background jobs to finish
wait
