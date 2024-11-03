#!/bin/bash
set -x  # Enable debug mode

# Number of logical cores to assign per process
LOGICAL_CORES=8  # 8 logical cores per experiment
DATE="2024-11-03"  # Experiment date as a variable
EXE_FILE="sphere/bo_parafac.py"  # Experiment file to run

# Create necessary directories if they don't exist
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/
mkdir -p temp/

# Parameters
ITER=500  # Number of iterations for ParafacSampler
SEED_START=0  # Starting seed value
SEED_END=4  # Ending seed value (5 seeds in total)
DIMENSIONS=(2 3 5 7)
CP_RANKS=(2)
# CP_MASK_RATIOS=(0 0.1)
CP_MASK_RATIOS=(0.1)
TRADE_OFF_PARAMS=(3)
CP_RANDOM_DIST_TYPE="uniform"  # Distribution type for random sampling

# Define initial CPU core range for taskset
CPU_CORE_START=0
TOTAL_CORES=$(nproc)  # Get total available cores
MAX_CONCURRENT_JOBS=$((TOTAL_CORES / LOGICAL_CORES))  # Calculate max concurrent jobs

# Counter for active processes
active_jobs=0

# Loop through dimensions, cp_rank, cp_mask_ratio, trade_off_param, and seeds
for DIM in "${DIMENSIONS[@]}"; do
    for CP_RANK in "${CP_RANKS[@]}"; do
        for CP_MASK_RATIO in "${CP_MASK_RATIOS[@]}"; do
            for TRADE_OFF_PARAM in "${TRADE_OFF_PARAMS[@]}"; do
                for SEED in $(seq $SEED_START $SEED_END); do

                    # Set up experiment name base
                    EXPERIMENT_NAME_BASE="benchmark_parafac_dim${DIM}_rank${CP_RANK}_mask${CP_MASK_RATIO}_tradeoff${TRADE_OFF_PARAM}_seed${SEED}"
                    
                    # If CP_MASK_RATIO is 0, only run with include_observed_points=False
                    if (( $(echo "$CP_MASK_RATIO == 0" | bc -l) )); then
                        INCLUDE_OBSERVED=False
                        EXPERIMENT_NAME="${EXPERIMENT_NAME_BASE}_includeObs${INCLUDE_OBSERVED}"
                        LOG_FILE="temp/${EXPERIMENT_NAME}.log"
                        
                        # Calculate core range for this job
                        CORE_RANGE="$((CPU_CORE_START))-$((CPU_CORE_START + LOGICAL_CORES - 1))"
                        CMD="taskset -c $CORE_RANGE python3 experiments/${DATE}/${EXE_FILE} \
                            --dimensions $DIM \
                            --cp_rank $CP_RANK \
                            --cp_mask_ratio $CP_MASK_RATIO \
                            --acq_trade_off_param $TRADE_OFF_PARAM \
                            --seed $SEED \
                            --iter_bo $ITER \
                            --cp_random_dist_type $CP_RANDOM_DIST_TYPE"
                        
                        # Execute command in background
                        echo "Running experiment with mask_ratio 0, include_observed_points=False on cores $CORE_RANGE"
                        $CMD > "$LOG_FILE" 2>&1 &
                        
                        # Update CPU_CORE_START and active_jobs count
                        CPU_CORE_START=$(( (CPU_CORE_START + LOGICAL_CORES) % TOTAL_CORES ))
                        active_jobs=$((active_jobs + 1))
                        
                    else
                        for INCLUDE_OBSERVED in False True; do
                            EXPERIMENT_NAME="${EXPERIMENT_NAME_BASE}_includeObs${INCLUDE_OBSERVED}"
                            LOG_FILE="temp/${EXPERIMENT_NAME}.log"

                            # Calculate core range for this job
                            CORE_RANGE="$((CPU_CORE_START))-$((CPU_CORE_START + LOGICAL_CORES - 1))"
                            CMD="taskset -c $CORE_RANGE python3 experiments/${DATE}/${EXE_FILE} \
                                --dimensions $DIM \
                                --cp_rank $CP_RANK \
                                --cp_mask_ratio $CP_MASK_RATIO \
                                --acq_trade_off_param $TRADE_OFF_PARAM \
                                --seed $SEED \
                                --iter_bo $ITER \
                                --cp_random_dist_type $CP_RANDOM_DIST_TYPE"
                            
                            # Add --include_observed_points if INCLUDE_OBSERVED is True
                            if [ "$INCLUDE_OBSERVED" == "True" ]; then
                                CMD+=" --include_observed_points"
                            fi
                            
                            # Execute command in background
                            echo "Running experiment with mask_ratio $CP_MASK_RATIO, include_observed_points=$INCLUDE_OBSERVED on cores $CORE_RANGE"
                            $CMD > "$LOG_FILE" 2>&1 &
                            
                            # Update CPU_CORE_START and active_jobs count
                            CPU_CORE_START=$(( (CPU_CORE_START + LOGICAL_CORES) % TOTAL_CORES ))
                            active_jobs=$((active_jobs + 1))
                        done
                    fi

                    # Check if active_jobs reached max limit
                    if (( active_jobs >= MAX_CONCURRENT_JOBS )); then
                        wait -n  # Wait for at least one job to finish
                        active_jobs=$((active_jobs - 1))
                    fi
                    
                    echo "Log saved to $LOG_FILE"
                done
            done
        done
    done
done

# Wait for all background jobs to complete
wait
