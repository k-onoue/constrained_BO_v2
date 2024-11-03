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
CP_MASK_RATIOS=(0 0.1)
TRADE_OFF_PARAMS=(3)
CP_RANDOM_DIST_TYPE="uniform"  # Distribution type for random sampling

# Define initial CPU core range for taskset
CPU_CORE_START=0
TOTAL_CORES=$(nproc)  # Get total available cores

# Loop through dimensions, cp_rank, cp_mask_ratio, trade_off_param, and seeds
for DIM in "${DIMENSIONS[@]}"; do
    for CP_RANK in "${CP_RANKS[@]}"; do
        for CP_MASK_RATIO in "${CP_MASK_RATIOS[@]}"; do
            for TRADE_OFF_PARAM in "${TRADE_OFF_PARAMS[@]}"; do
                for SEED in $(seq $SEED_START $SEED_END); do

                    # Set up experiment name base
                    EXPERIMENT_NAME_BASE="benchmark_parafac_dim${DIM}_rank${CP_RANK}_mask${CP_MASK_RATIO}_tradeoff${TRADE_OFF_PARAM}_seed${SEED}"
                    
                    # Check if CP_MASK_RATIO is 0 and run only with include_observed_points=False
                    if (( $(echo "$CP_MASK_RATIO == 0" | bc -l) )); then
                        INCLUDE_OBSERVED=False
                        EXPERIMENT_NAME="${EXPERIMENT_NAME_BASE}_includeObs${INCLUDE_OBSERVED}"
                        LOG_FILE="temp/${EXPERIMENT_NAME}.log"
                        
                        # Base command
                        CMD="python3 experiments/${DATE}/${EXE_FILE} \
                            --dimensions $DIM \
                            --cp_rank $CP_RANK \
                            --cp_mask_ratio $CP_MASK_RATIO \
                            --acq_trade_off_param $TRADE_OFF_PARAM \
                            --seed $SEED \
                            --iter_bo $ITER \
                            --cp_random_dist_type $CP_RANDOM_DIST_TYPE"
                        
                        echo "Running experiment with mask_ratio 0, include_observed_points=False"
                        $CMD > "$LOG_FILE" 2>&1

                    else
                        # Run both include_observed_points=False and include_observed_points=True
                        for INCLUDE_OBSERVED in False True; do
                            EXPERIMENT_NAME="${EXPERIMENT_NAME_BASE}_includeObs${INCLUDE_OBSERVED}"
                            LOG_FILE="temp/${EXPERIMENT_NAME}.log"

                            # Base command
                            CMD="python3 experiments/${DATE}/${EXE_FILE} \
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
                            
                            echo "Running experiment with mask_ratio $CP_MASK_RATIO, include_observed_points=$INCLUDE_OBSERVED"
                            $CMD > "$LOG_FILE" 2>&1
                        done
                    fi

                    echo "Log saved to $LOG_FILE"
                done
            done
        done
    done
done

# Wait for all background jobs to complete
wait
