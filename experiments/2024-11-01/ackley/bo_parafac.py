import argparse
import logging
import os
from functools import partial

import numpy as np
import optuna
from _src import DB_DIR, LOG_DIR, ParafacSampler, set_logger


def ackley(x):
    """
    Computes the d-dimensional Ackley function.
    :param x: np.array, shape (d,) - point at which to evaluate the function.
    :return: float - value of the Ackley function at x.
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    sum2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return sum1 + sum2 + a + np.exp(1)


def objective(trial, dimensions):
    """
    Objective function for Bayesian optimization using the d-dimensional Ackley function.
    """
    # Suggest integer-valued parameters in each dimension within the range [-5, 5]
    x = np.array([trial.suggest_int(f"x_{i}", -5, 5) for i in range(dimensions)])
    
    # Compute the Ackley function
    return ackley(x)


def run_bo(settings):
    """
    Run the Bayesian optimization experiment using the specified settings.
    """
    optuna.logging.set_verbosity(optuna.logging.DEBUG)



    dimensions = settings["dimensions"]  # Number of dimensions for the Ackley function
    
    # Set up the ParafacSampler for Bayesian optimization
    sampler = ParafacSampler(
        cp_rank=settings["cp_settings"]["rank"],  # Rank for the CP decomposition
        als_iter_num=settings["cp_settings"]["als_iterations"],  # ALS iterations
        mask_ratio=settings["cp_settings"]["mask_ratio"],  # Mask ratio for CP decomposition
        trade_off_param=settings["acqf_settings"]["trade_off_param"],
        distribution_type=settings["cp_settings"]["random_dist_type"],  # Distribution type
        seed=settings["seed"],  # Random seed for reproducibility
        unique_sampling=settings["unique_sampling"],  # Apply the unique_sampling flag
    )

    # Determine whether to minimize or maximize based on acq_maximize flag
    direction = "maximize" if settings["acqf_settings"]["maximize"] else "minimize"

    # Create or load the study
    try:
        study = optuna.load_study(
            study_name=settings["name"], storage=settings["storage"]
        )
        logging.info(f"Resuming study '{settings['name']}' from {settings['storage']}")
    except KeyError:
        study = optuna.create_study(
            study_name=settings["name"],
            sampler=sampler,
            direction=direction,
            storage=settings["storage"],
        )
        logging.info(f"Created new study '{settings['name']}' in {settings['storage']}")

    # Bind the dimension to the objective function
    objective_with_args = partial(objective, dimensions=dimensions)

    # Optimize
    study.optimize(objective_with_args, n_trials=settings["iter_bo"])

    # Log final results
    logging.info(f"Best value: {study.best_value}")
    logging.info(f"Best params: {study.best_params}")


def parse_args():
    """
    Parse command-line arguments to specify experiment settings.
    """
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization Experiment with ParafacSampler for Ackley function"
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--iter_bo",
        type=int,
        default=300,
        help="Number of iterations for Bayesian optimization.",
    )
    parser.add_argument(
        "--cp_rank", type=int, default=2, help="Rank for the CP decomposition."
    )
    parser.add_argument(
        "--cp_als_iterations",
        type=int,
        default=100,
        help="Number of ALS iterations for the CP decomposition.",
    )
    parser.add_argument(
        "--cp_mask_ratio",
        type=float,
        default=0.1,
        help="Mask ratio used in the CP decomposition.",
    )
    parser.add_argument(
        "--cp_random_dist_type",
        type=str,
        choices=["uniform", "normal"],
        default="uniform",
        help="Distribution type for random sampling.",
    )
    parser.add_argument(
        "--acq_trade_off_param",
        type=float,
        default=3.0,
        help="Trade-off parameter for the acquisition function.",
    )
    parser.add_argument(
        "--acq_maximize",
        action="store_true",
        help="Whether to maximize the acquisition function.",
    )
    parser.add_argument(
        "--unique_sampling",
        action="store_true",
        help="Whether to use unique sampling in the ParafacSampler.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=2,
        help="Number of dimensions for the Ackley function.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Get the script name to use for logging and experiment identification
    base_script_name = os.path.splitext(__file__.split("/")[-1])[0]

    # Parse the command-line arguments
    args = parse_args()

    # Concatenate parameters to create a unique script name for each configuration
    script_name = (
        f"{base_script_name}_dim{args.dimensions}_rank{args.cp_rank}_"
        f"mask{args.cp_mask_ratio}_tradeoff{args.acq_trade_off_param}_seed{args.seed}"
    )

    # Set up logging and retrieve the log filename
    log_filename = set_logger(script_name, LOG_DIR)

    # Use log_filename as the storage name in DB_DIR
    storage_filename = os.path.splitext(log_filename)[0] + ".db"
    storage_path = os.path.join(DB_DIR, storage_filename)
    storage_url = f"sqlite:///{storage_path}"

    # Define the experimental settings using argparse inputs
    settings = {
        "name": script_name,
        "seed": args.seed,
        "dimensions": args.dimensions,
        "iter_bo": args.iter_bo,  # Number of iterations for Bayesian optimization
        "storage": storage_url,  # Full path for the SQLite database in DB_DIR
        "unique_sampling": args.unique_sampling,  # Apply the unique_sampling flag
        "cp_settings": {
            "rank": args.cp_rank,  # Rank for the CP decomposition
            "als_iterations": args.cp_als_iterations,  # ALS iterations for the CP decomposition
            "mask_ratio": args.cp_mask_ratio,  # Mask ratio used in the CP decomposition
            "random_dist_type": args.cp_random_dist_type,  # Distribution type for random sampling
        },
        "acqf_settings": {
            "trade_off_param": args.acq_trade_off_param,  # Trade-off parameter for acquisition function
            "maximize": args.acq_maximize,  # Whether to maximize the acquisition function
        },
    }

    logging.info(f"Experiment settings: {settings}")

    # Run the Bayesian optimization experiment
    run_bo(settings)


# ackley_1
"""
#!/bin/bash

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/
mkdir -p temp/

# Parameters
ITER=500  # Number of iterations for ParafacSampler
SEED_START=0  # Starting seed value
SEED_END=4  # Ending seed value (5 seeds in total)
TEMP="temp"  # Temporary directory for log files

# Define the list of dimensions, cp_rank, cp_mask_ratio, and trade_off_param values
DIMENSIONS=(2 3 5 7)
CP_RANKS=(1 2)
CP_MASK_RATIOS=(0.1 0.2 0.33)
TRADE_OFF_PARAMS=(1 3 5)


# Loop through dimensions, cp_rank, cp_mask_ratio, trade_off_param, and seeds
for DIM in "${DIMENSIONS[@]}"; do
    for CP_RANK in "${CP_RANKS[@]}"; do
        for CP_MASK_RATIO in "${CP_MASK_RATIOS[@]}"; do
            for TRADE_OFF_PARAM in "${TRADE_OFF_PARAMS[@]}"; do
                for SEED in $(seq $SEED_START $SEED_END); do

                    # Set up experiment name and log file paths
                    EXPERIMENT_NAME="benchmark_parafac_dim${DIM}_rank${CP_RANK}_mask${CP_MASK_RATIO}_tradeoff${TRADE_OFF_PARAM}_seed${SEED}"
                    LOG_FILE="${TEMP}/${EXPERIMENT_NAME}.log"

                    echo "Running experiment with ParafacSampler, dimension $DIM, cp_rank $CP_RANK, mask_ratio $CP_MASK_RATIO, trade_off_param $TRADE_OFF_PARAM, seed $SEED..."

                    # Run each experiment and log the output
                    python3 experiments/2024-10-25/ackley/bo_parafac.py \
                        --dimensions $DIM \
                        --cp_rank $CP_RANK \
                        --cp_mask_ratio $CP_MASK_RATIO \
                        --acq_trade_off_param $TRADE_OFF_PARAM \
                        --seed $SEED \
                        --iter_bo $ITER \
                        > "$LOG_FILE" 2>&1

                    echo "Log saved to $LOG_FILE"
                done
            done
        done
    done
done
"""

# ackley_2: 重複サンプリングを許さない
"""
#!/bin/bash

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/
mkdir -p temp/

# Parameters
ITER=500  # Number of iterations for ParafacSampler
SEED_START=0  # Starting seed value
SEED_END=4  # Ending seed value (5 seeds in total)
TEMP="temp"  # Temporary directory for log files

# Define the list of dimensions, cp_rank, cp_mask_ratio, and trade_off_param values
DIMENSIONS=(2 3 5 7)
CP_RANKS=(1 2)
CP_MASK_RATIOS=(0.1 0.2 0.33)
TRADE_OFF_PARAMS=(1 3 5)


# Loop through dimensions, cp_rank, cp_mask_ratio, trade_off_param, and seeds
for DIM in "${DIMENSIONS[@]}"; do
    for CP_RANK in "${CP_RANKS[@]}"; do
        for CP_MASK_RATIO in "${CP_MASK_RATIOS[@]}"; do
            for TRADE_OFF_PARAM in "${TRADE_OFF_PARAMS[@]}"; do
                for SEED in $(seq $SEED_START $SEED_END); do

                    # Set up experiment name and log file paths
                    EXPERIMENT_NAME="benchmark_parafac_dim${DIM}_rank${CP_RANK}_mask${CP_MASK_RATIO}_tradeoff${TRADE_OFF_PARAM}_seed${SEED}"
                    LOG_FILE="${TEMP}/${EXPERIMENT_NAME}.log"

                    echo "Running experiment with ParafacSampler, dimension $DIM, cp_rank $CP_RANK, mask_ratio $CP_MASK_RATIO, trade_off_param $TRADE_OFF_PARAM, seed $SEED..."

                    # Run each experiment and log the output
                    python3 experiments/2024-10-25/ackley/bo_parafac.py \
                        --dimensions $DIM \
                        --cp_rank $CP_RANK \
                        --cp_mask_ratio $CP_MASK_RATIO \
                        --acq_trade_off_param $TRADE_OFF_PARAM \
                        --seed $SEED \
                        --iter_bo $ITER \
                        --unique_sampling \
                        > "$LOG_FILE" 2>&1

                    echo "Log saved to $LOG_FILE"
                done
            done
        done
    done
done
"""

# ackley_3: cp ランクと性能 
"""
#!/bin/bash

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/
mkdir -p temp/

# Parameters
ITER=500  # Number of iterations for ParafacSampler
SEED_START=0  # Starting seed value
SEED_END=4  # Ending seed value (5 seeds in total)
TEMP="temp"  # Temporary directory for log files

# Define the list of dimensions, cp_rank, cp_mask_ratio, and trade_off_param values
DIMENSIONS=(2 3 5 7)
CP_RANKS=(1 2 3 4)
CP_MASK_RATIOS=(0.33)
TRADE_OFF_PARAMS=(3)


# Loop through dimensions, cp_rank, cp_mask_ratio, trade_off_param, and seeds
for DIM in "${DIMENSIONS[@]}"; do
    for CP_RANK in "${CP_RANKS[@]}"; do
        for CP_MASK_RATIO in "${CP_MASK_RATIOS[@]}"; do
            for TRADE_OFF_PARAM in "${TRADE_OFF_PARAMS[@]}"; do
                for SEED in $(seq $SEED_START $SEED_END); do

                    # Set up experiment name and log file paths
                    EXPERIMENT_NAME="benchmark_parafac_dim${DIM}_rank${CP_RANK}_mask${CP_MASK_RATIO}_tradeoff${TRADE_OFF_PARAM}_seed${SEED}"
                    LOG_FILE="${TEMP}/${EXPERIMENT_NAME}.log"

                    echo "Running experiment with ParafacSampler, dimension $DIM, cp_rank $CP_RANK, mask_ratio $CP_MASK_RATIO, trade_off_param $TRADE_OFF_PARAM, seed $SEED..."

                    # Run each experiment and log the output
                    python3 experiments/2024-10-25/ackley/bo_parafac.py \
                        --dimensions $DIM \
                        --cp_rank $CP_RANK \
                        --cp_mask_ratio $CP_MASK_RATIO \
                        --acq_trade_off_param $TRADE_OFF_PARAM \
                        --seed $SEED \
                        --iter_bo $ITER \
                        > "$LOG_FILE" 2>&1

                    echo "Log saved to $LOG_FILE"
                done
            done
        done
    done
done
"""

# ackley_4: normal distribution
"""
#!/bin/bash

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/
mkdir -p temp/

# Parameters
ITER=500  # Number of iterations for ParafacSampler
SEED_START=0  # Starting seed value
SEED_END=4  # Ending seed value (5 seeds in total)
TEMP="temp"  # Temporary directory for log files

# Define the list of dimensions, cp_rank, cp_mask_ratio, trade_off_param values, and distribution type
DIMENSIONS=(2 3 5 7)
CP_RANKS=(1 2 3 4)
CP_MASK_RATIOS=(0.33)
TRADE_OFF_PARAMS=(3)
CP_RANDOM_DIST_TYPE="normal"  # Distribution type for random sampling

# Loop through dimensions, cp_rank, cp_mask_ratio, trade_off_param, and seeds
for DIM in "${DIMENSIONS[@]}"; do
    for CP_RANK in "${CP_RANKS[@]}"; do
        for CP_MASK_RATIO in "${CP_MASK_RATIOS[@]}"; do
            for TRADE_OFF_PARAM in "${TRADE_OFF_PARAMS[@]}"; do
                for SEED in $(seq $SEED_START $SEED_END); do

                    # Set up experiment name and log file paths
                    EXPERIMENT_NAME="benchmark_parafac_dim${DIM}_rank${CP_RANK}_mask${CP_MASK_RATIO}_tradeoff${TRADE_OFF_PARAM}_seed${SEED}"
                    LOG_FILE="${TEMP}/${EXPERIMENT_NAME}.log"

                    echo "Running experiment with ParafacSampler, dimension $DIM, cp_rank $CP_RANK, mask_ratio $CP_MASK_RATIO, trade_off_param $TRADE_OFF_PARAM, seed $SEED..."

                    # Run each experiment and log the output
                    python3 experiments/2024-10-25/ackley/bo_parafac.py \
                        --dimensions $DIM \
                        --cp_rank $CP_RANK \
                        --cp_mask_ratio $CP_MASK_RATIO \
                        --acq_trade_off_param $TRADE_OFF_PARAM \
                        --seed $SEED \
                        --iter_bo $ITER \
                        --cp_random_dist_type $CP_RANDOM_DIST_TYPE \
                        > "$LOG_FILE" 2>&1

                    echo "Log saved to $LOG_FILE"
                done
            done
        done
    done
done

"""