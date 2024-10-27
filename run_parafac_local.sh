#!/bin/bash
set -x  # Enable debug mode

# Number of logical cores to assign per process
LOGICAL_CORES=8  # 8 logical cores per experiment
DATE="2024-10-26"  # Experiment date as a variable

# Create necessary directories if they don't exist
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/
mkdir -p temp/

# Parameters
ITER=3000  # Number of iterations for ParafacSampler
SEED_START=0  # Starting seed value
SEED_END=4  # Ending seed value (5 seeds in total)
DIMENSIONS=(2 3 5 7)
CP_RANKS=(2)
CP_MASK_RATIOS=(0.33)
TRADE_OFF_PARAMS=(3)
CP_RANDOM_DIST_TYPE="normal"  # Distribution type for random sampling

# Define initial CPU core range for taskset
CPU_CORE_START=0
TOTAL_CORES=$(nproc)  # Get total available cores

# Loop through dimensions, cp_rank, cp_mask_ratio, trade_off_param, and seeds
for DIM in "${DIMENSIONS[@]}"; do
    for CP_RANK in "${CP_RANKS[@]}"; do
        for CP_MASK_RATIO in "${CP_MASK_RATIOS[@]}"; do
            for TRADE_OFF_PARAM in "${TRADE_OFF_PARAMS[@]}"; do
                for SEED in $(seq $SEED_START $SEED_END); do

                    # Set up experiment name and log file paths
                    EXPERIMENT_NAME="benchmark_parafac_dim${DIM}_rank${CP_RANK}_mask${CP_MASK_RATIO}_tradeoff${TRADE_OFF_PARAM}_seed${SEED}"
                    LOG_FILE="temp/${EXPERIMENT_NAME}.log"
                    
                    # Check if sufficient cores are available for parallel execution
                    if [ $((CPU_CORE_START + LOGICAL_CORES)) -le $TOTAL_CORES ]; then
                        # Run in parallel using taskset if enough cores are available
                        echo "Running experiment in parallel: dimension $DIM, cp_rank $CP_RANK, mask_ratio $CP_MASK_RATIO, trade_off_param $TRADE_OFF_PARAM, seed $SEED on cores $CPU_CORE_START-$((CPU_CORE_START + LOGICAL_CORES - 1))"
                        
                        taskset -c $CPU_CORE_START-$((CPU_CORE_START + LOGICAL_CORES - 1)) \
                        python3 experiments/${DATE}/sphere/bo_parafac.py \
                            --dimensions $DIM \
                            --cp_rank $CP_RANK \
                            --cp_mask_ratio $CP_MASK_RATIO \
                            --acq_trade_off_param $TRADE_OFF_PARAM \
                            --seed $SEED \
                            --iter_bo $ITER \
                            --cp_random_dist_type $CP_RANDOM_DIST_TYPE \
                            > "$LOG_FILE" 2>&1 &
                        
                        # Update CPU core range for the next process
                        CPU_CORE_START=$((CPU_CORE_START + LOGICAL_CORES))
                    else
                        # Run without taskset if not enough cores are available
                        echo "Running experiment without parallelization: dimension $DIM, cp_rank $CP_RANK, mask_ratio $CP_MASK_RATIO, trade_off_param $TRADE_OFF_PARAM, seed $SEED"
                        
                        python3 experiments/${DATE}/sphere/bo_parafac.py \
                            --dimensions $DIM \
                            --cp_rank $CP_RANK \
                            --cp_mask_ratio $CP_MASK_RATIO \
                            --acq_trade_off_param $TRADE_OFF_PARAM \
                            --seed $SEED \
                            --iter_bo $ITER \
                            --cp_random_dist_type $CP_RANDOM_DIST_TYPE \
                            > "$LOG_FILE" 2>&1
                    fi

                    # Reset CPU core start if it exceeds the total cores
                    if [ $CPU_CORE_START -ge $TOTAL_CORES ]; then
                        CPU_CORE_START=0
                    fi

                    echo "Log saved to $LOG_FILE"
                done
            done
        done
    done
done

# Wait for all background jobs to complete
wait





# #!/bin/bash
# set -x  # Enable debug mode

# # Create results and logs directories if they don't exist
# mkdir -p results/
# mkdir -p results/logs/
# mkdir -p results/dbs/
# mkdir -p temp/

# # Parameters
# ITER=500  # Number of iterations for ParafacSampler
# SEED_START=0  # Starting seed value
# SEED_END=4  # Ending seed value (5 seeds in total)
# TEMP="temp"  # Temporary directory for log files

# # Define the list of dimensions, cp_rank, cp_mask_ratio, trade_off_param values, and distribution type
# DIMENSIONS=(2 3 5 7)
# CP_RANKS=(1 2 3 4)
# CP_MASK_RATIOS=(0.33)
# TRADE_OFF_PARAMS=(3)
# CP_RANDOM_DIST_TYPE="normal"  # Distribution type for random sampling

# # Loop through dimensions, cp_rank, cp_mask_ratio, trade_off_param, and seeds
# for DIM in "${DIMENSIONS[@]}"; do
#     for CP_RANK in "${CP_RANKS[@]}"; do
#         for CP_MASK_RATIO in "${CP_MASK_RATIOS[@]}"; do
#             for TRADE_OFF_PARAM in "${TRADE_OFF_PARAMS[@]}"; do
#                 for SEED in $(seq $SEED_START $SEED_END); do

#                     # Set up experiment name and log file paths
#                     EXPERIMENT_NAME="benchmark_parafac_dim${DIM}_rank${CP_RANK}_mask${CP_MASK_RATIO}_tradeoff${TRADE_OFF_PARAM}_seed${SEED}"
#                     LOG_FILE="${TEMP}/${EXPERIMENT_NAME}.log"

#                     echo "Running experiment with ParafacSampler, dimension $DIM, cp_rank $CP_RANK, mask_ratio $CP_MASK_RATIO, trade_off_param $TRADE_OFF_PARAM, seed $SEED..."

#                     # Run each experiment and log the output
#                     python3 experiments/2024-10-25/ackley/bo_parafac.py \
#                         --dimensions $DIM \
#                         --cp_rank $CP_RANK \
#                         --cp_mask_ratio $CP_MASK_RATIO \
#                         --acq_trade_off_param $TRADE_OFF_PARAM \
#                         --seed $SEED \
#                         --iter_bo $ITER \
#                         --cp_random_dist_type $CP_RANDOM_DIST_TYPE \
#                         > "$LOG_FILE" 2>&1

#                     echo "Log saved to $LOG_FILE"
#                 done
#             done
#         done
#     done
# done
