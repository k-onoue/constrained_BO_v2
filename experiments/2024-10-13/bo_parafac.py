import argparse
import logging
import os
from functools import partial

import numpy as np
import optuna
from _src import DB_DIR, LOG_DIR, ParafacSampler, WarcraftObjective, set_logger


def objective(trial, map_shape, objective_function):
    """
    Objective function for the Bayesian optimization process.
    """
    directions = ["oo", "ab", "ac", "ad", "bc", "bd", "cd"]
    x = np.empty(map_shape, dtype=object)

    # Suggest categorical parameters for each cell in the matrix
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            x[i, j] = trial.suggest_categorical(f"x_{i}_{j}", directions)

    # Evaluate the objective function using the suggested directions
    return objective_function(x)


def run_bo(settings):
    """
    Run the Bayesian optimization experiment using the specified settings.
    """
    # Set up the target map and objective function
    map_targeted = settings["map"]
    map_shape = map_targeted.shape
    objective_function = WarcraftObjective(map_targeted)

    # Set up the sampler for Bayesian optimization, applying the unique_sampling flag
    sampler = ParafacSampler(
        trade_off_param=settings["acqf_settings"]["trade_off_param"],
        unique_sampling=settings["unique_sampling"],  # Apply the flag from settings
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

    # Use functools.partial to bind additional arguments
    objective_with_args = partial(
        objective, map_shape=map_shape, objective_function=objective_function
    )

    study.optimize(objective_with_args, n_trials=settings["iter_bo"])

    # Log final results
    best_x = np.empty(map_shape, dtype=object)
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            best_x[i, j] = study.best_params[f"x_{i}_{j}"]

    logging.info(f"Best value: {study.best_value}")
    logging.info(f"Best params: {study.best_params}")
    logging.info(f"Best Direction Matrix:\n{best_x}")

    # Optionally visualize the optimization history
    optuna.visualization.plot_optimization_history(study)


def parse_args():
    """
    Parse command-line arguments to specify experiment settings.
    """
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization Experiment for Warcraft Pathfinding"
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
        "--acq_batch_size", type=int, default=10, help="Batch size for optimization."
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

    return parser.parse_args()


if __name__ == "__main__":
    # Get the script name to use for logging and experiment identification
    base_script_name = os.path.splitext(__file__.split("/")[-1])[0]

    # Parse the command-line arguments
    args = parse_args()

    # Define the target map for optimization
    map_targeted = np.array([[1, 4], [2, 1]])
    map_targeted = map_targeted / map_targeted.sum()

    # Concatenate the sampler type to the script name
    script_name = f"{base_script_name}_seed{args.seed}"

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
        "map": map_targeted,
        "iter_bo": args.iter_bo,  # Number of iterations for Bayesian optimization
        "storage": storage_url,  # Full path for the SQLite database in DB_DIR
        "unique_sampling": args.unique_sampling,  # Apply the unique_sampling flag
        "cp_settings": {
            "dim": len(
                map_targeted.flatten()
            ),  # Dimensionality of the CP decomposition
            "rank": args.cp_rank,  # Rank for the CP decomposition
            "als_iterations": args.cp_als_iterations,  # ALS iterations for the CP decomposition
            "mask_ratio": args.cp_mask_ratio,  # Mask ratio used in the CP decomposition
            "random_dist_type": args.cp_random_dist_type,  # Distribution type for random sampling
        },
        "acqf_settings": {
            "trade_off_param": args.acq_trade_off_param,  # Trade-off parameter for acquisition function
            "batch_size": args.acq_batch_size,  # Batch size for optimization
            "maximize": args.acq_maximize,  # Whether to maximize the acquisition function
        },
    }

    logging.info(f"Experiment settings: {settings}")

    # Run the Bayesian optimization experiment
    run_bo(settings)
