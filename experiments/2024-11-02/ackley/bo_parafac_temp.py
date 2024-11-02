import argparse
import logging
import os
from functools import partial

import numpy as np
import optuna
from _src import DB_DIR, LOG_DIR, set_logger


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


import logging
from typing import Literal

import numpy as np
import optuna
from optuna.samplers import BaseSampler
from tensorly import cp_to_tensor
from tensorly.decomposition import parafac
from tensorly.decomposition import non_negative_parafac


class ParafacSampler(BaseSampler):
    def __init__(
        self,
        cp_rank: int = 3,
        als_iter_num: int = 10,
        mask_ratio: float = 0.2,
        trade_off_param: float = 1.0,
        distribution_type: Literal["uniform", "normal"] = "uniform",
        seed: int = None,
        unique_sampling: bool = False,
    ):
        """
        Initialize the ParafacSampler with the necessary settings.

        Parameters:
        - cp_rank: int, the rank of the CP decomposition.
        - als_iter_num: int, the number of ALS iterations to perform during decomposition.
        - mask_ratio: float, the ratio to control the number of masks for CP decomposition.
        - trade_off_param: float, the trade-off parameter between exploration and exploitation.
        - distribution_type: str, "uniform" or "normal" to choose the distribution type.
        - seed: int, random seed.
        """
        self.cp_rank = cp_rank
        self.als_iter_num = als_iter_num
        self.mask_ratio = mask_ratio
        self.tensor_decomposition_setting = {
            "cp_rank": cp_rank,
            "als_iter_num": als_iter_num,
            "mask_ratio": mask_ratio,
        }
        self.trade_off_param = trade_off_param
        self.distribution_type = distribution_type
        self.rng = np.random.RandomState(seed)
        self.unique_sampling = unique_sampling
        # Internal storage
        self._param_names = None
        self._category_maps = None  # Map from param name to list of categories
        self._shape = None  # Shape of the tensor
        self._evaluated_indices = []  # List of indices of evaluated points
        self._tensor_eval = None  # Tensor of evaluations
        self._tensor_eval_bool = None  # Mask tensor indicating evaluated points
        self._maximize = None

    def infer_relative_search_space(self, study, trial):
        search_space = optuna.search_space.intersection_search_space(
            study.get_trials(deepcopy=False)
        )

        print()
        print(search_space)
        print()

        # Include integer and categorical distributions
        relevant_search_space = {}
        for name, distribution in search_space.items():

            # IntDistribution should be bounded
            #########################################################
            #########################################################
            #########################################################
            #########################################################
            #########################################################
            #########################################################
            #########################################################
            #########################################################
            if isinstance(distribution, (optuna.distributions.IntDistribution,
                                         optuna.distributions.CategoricalDistribution)):
                relevant_search_space[name] = distribution
        return relevant_search_space

    def sample_relative(self, study, trial, search_space):
        logging.info("Using sample_relative for sampling.")

        if not search_space:
            return {}

        # Initialize internal structures if not already done
        if self._param_names is None:
            self._param_names = list(search_space.keys())
            self._param_names.sort()  # Ensure consistent order
            self._category_maps = {}
            self._shape = []
            for param_name in self._param_names:
                distribution = search_space[param_name]
                if isinstance(distribution, optuna.distributions.CategoricalDistribution):
                    categories = distribution.choices

                #########################################################
                #########################################################
                #########################################################
                #########################################################
                #########################################################
                #########################################################
                # IntDistribution should be bounded
                elif isinstance(distribution, optuna.distributions.IntDistribution):
                    # Treat integers as categorical by creating a list of possible values
                    categories = list(range(distribution.low, distribution.high + 1, distribution.step))
                else:
                    # Skip unsupported distributions
                    continue
                self._category_maps[param_name] = categories
                self._shape.append(len(categories))
            self._shape = tuple(self._shape)
            # Initialize tensors
            self._tensor_eval = np.full(self._shape, np.nan)
            self._tensor_eval_bool = np.zeros(self._shape, dtype=bool)
            self._evaluated_indices = []
            self._maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE

        # Build tensor from past trials
        self._update_tensor(study)

        # Perform CP decomposition and suggest next parameter set
        mean_tensor, std_tensor = self._fit(
            self._tensor_eval,
            self._tensor_eval_bool,
            self._evaluated_indices,
            self.distribution_type,
        )

        # Suggest next indices based on UCB
        next_indices = self._suggest_ucb_candidates(
            mean_tensor=mean_tensor,
            std_tensor=std_tensor,
            trade_off_param=self.trade_off_param,
            batch_size=1,  # We only need one sample here
            maximize=self._maximize,
        )

        # Convert indices back to parameter values
        next_index = next_indices[0]
        params = {}
        for i, param_name in enumerate(self._param_names):
            category_index = next_index[i]
            category = self._category_maps[param_name][category_index]
            params[param_name] = category

        return params

    def sample_independent(self, study, trial, param_name, param_distribution):

        logging.info(f"Sampled independent: {param_name}, {param_distribution}")


        # Fallback to random sampling
        return optuna.samplers.RandomSampler().sample_independent(
            study, trial, param_name, param_distribution
        )
    

    def _update_tensor(self, study):
        # Go through all completed trials and update the tensor
        trials = study.get_trials(deepcopy=False)
        for trial in trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            # Get parameter indices
            index = []
            for param_name in self._param_names:
                if param_name not in trial.params:
                    break  # Skip trials that don't have all parameters
                category = trial.params[param_name]
                try:
                    category_index = self._category_maps[param_name].index(category)
                except ValueError:
                    break  # Skip trials with unknown category
                index.append(category_index)
            else:
                index = tuple(index)
                if index not in self._evaluated_indices:
                    value = trial.value
                    self._tensor_eval[index] = value
                    self._tensor_eval_bool[index] = True
                    self._evaluated_indices.append(index)

    def _fit(
        self,
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray,
        all_evaluated_indices: list[tuple[int, ...]],
        distribution_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform CP decomposition and return the mean and variance tensors.
        """
        div = int(1 / self.mask_ratio)
        mask_split_list = self._split_list_equally(all_evaluated_indices, div)

        tensors_list = []

        eval_min = np.nanmin(tensor_eval[np.isfinite(tensor_eval)]) # Avoid NaN
        eval_max = np.nanmax(tensor_eval[np.isfinite(tensor_eval)]) # Avoid NaN
        eval_mean = np.nanmean(tensor_eval[np.isfinite(tensor_eval)])
        eval_std = np.nanstd(tensor_eval[np.isfinite(tensor_eval)])

        # Generate initial tensor with random values
        init_tensor_eval = self._generate_random_array(
            low=eval_min, 
            high=eval_max, 
            shape=tensor_eval.shape,
            distribution_type=distribution_type,
            mean=eval_mean,
            std_dev=eval_std,
        )

        init_tensor_eval[tensor_eval_bool == True] = tensor_eval[tensor_eval_bool == True]

        for mask_list in mask_split_list:
            mask_tensor = np.ones_like(tensor_eval_bool)
            for mask_index in mask_list:
                mask_tensor[mask_index] = False

            # print()
            # print(init_tensor_eval)
            # print()

            # Perform CP decomposition
            cp_tensor = parafac(
                init_tensor_eval,
                rank=self.cp_rank,
                mask=mask_tensor,
                n_iter_max=self.als_iter_num,
                init="random",
                random_state=self.rng,
            )

            # Convert the CP decomposition back to a tensor
            reconstructed_tensor = cp_to_tensor(cp_tensor)

            # Append the reconstructed tensor to the list for later processing
            tensors_list.append(reconstructed_tensor)

        # Calculate mean and variance tensors
        tensors_stack = np.stack(tensors_list)
        mean_tensor = np.mean(tensors_stack, axis=0)
        std_tensor = np.std(tensors_stack, axis=0)

        # Replace the mean and variance of known points with the original values and zeros
        mean_tensor[tensor_eval_bool == True] = tensor_eval[tensor_eval_bool == True]
        std_tensor[tensor_eval_bool == True] = 0

        return mean_tensor, std_tensor

    def _suggest_ucb_candidates(
        self,
        mean_tensor: np.ndarray,
        std_tensor: np.ndarray,
        trade_off_param: float,
        batch_size: int,
        maximize: bool,
    ) -> list[tuple[int, ...]]:
        """
        Suggest candidate points based on UCB values, selecting the top batch_size points.

        Returns:
        - indices: list of tuples, the indices of the top batch_size points based on UCB.
        """

        def _ucb(mean_tensor, std_tensor, trade_off_param, maximize=True) -> np.ndarray:
            mean_tensor = mean_tensor if maximize else -mean_tensor
            ucb_values = mean_tensor + trade_off_param * std_tensor
            return ucb_values

        # Calculate the UCB values using the internal function
        ucb_values = _ucb(mean_tensor, std_tensor, trade_off_param, maximize)

        if self.unique_sampling:
            # Mask out already evaluated points
            ucb_values[self._tensor_eval_bool == True] = -np.inf

        # Flatten the tensor and get the indices of the top UCB values
        flat_indices = np.argsort(ucb_values.flatten())[::-1]  # Sort in descending order

        top_indices = np.unravel_index(flat_indices[:batch_size], ucb_values.shape)
        top_indices = list(zip(*top_indices))

        for index in top_indices:
            logging.info(
                f"UCB value at {index}: {ucb_values[index]}, Mean: {mean_tensor[index]}, Std: {std_tensor[index]}"
            )

        return top_indices

    def _split_list_equally(
        self, input_list: list[tuple[int, ...]], div: int
    ) -> list[list[tuple[int, ...]]]:
        quotient, remainder = divmod(len(input_list), div)
        result = []
        start = 0
        for i in range(div):
            group_size = quotient + (1 if i < remainder else 0)
            result.append(input_list[start : start + group_size])
            start += group_size
        return result

    def _generate_random_array(
        self,
        low: float,
        high: float,
        shape: tuple[int, ...],
        distribution_type: str = "uniform",
        mean: float = 0,
        std_dev: float = 1,
    ) -> np.ndarray:
        """
        Generate an array of random numbers with specified bounds and distribution type.
        Adds small noise if the evaluated entries are identical to avoid singular matrix errors.
        """
        # Handle case where all entries are identical by adding small random noise
        if low == high:
            low = low - 1e-6
            high = high + 1e-6
            std_dev = std_dev + 1e-6

        if distribution_type == "uniform":
            # Generate uniform random numbers
            return self.rng.uniform(low, high, shape)
        elif distribution_type == "normal":
            # Generate normal random numbers and clip them to the specified bounds
            normal_random = self.rng.normal(mean, std_dev, shape)
            return np.clip(normal_random, low, high)
        else:
            raise ValueError("distribution_type must be either 'uniform' or 'normal'.")


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
        default=30,
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
    optuna.logging.set_verbosity(optuna.logging.DEBUG)


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