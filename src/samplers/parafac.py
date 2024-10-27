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

        logging.info(f"Using sample_independent for sampling.")

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


# class ParafacSampler(BaseSampler):
#     def __init__(
#         self,
#         cp_rank: int = 3,
#         als_iter_num: int = 10,
#         mask_ratio: float = 0.2,
#         trade_off_param: float = 1.0,
#         distribution_type: Literal["uniform", "normal"] = "uniform",
#         seed: int = None,
#         unique_sampling: bool = False,
#     ):
#         """
#         Initialize the ParafacSampler with the necessary settings.

#         Parameters:
#         - cp_rank: int, the rank of the CP decomposition.
#         - als_iter_num: int, the number of ALS iterations to perform during decomposition.
#         - mask_ratio: float, the ratio to control the number of masks for CP decomposition.
#         - trade_off_param: float, the trade-off parameter between exploration and exploitation.
#         - distribution_type: str, "uniform" or "normal" to choose the distribution type.
#         - seed: int, random seed.
#         """
#         self.cp_rank = cp_rank
#         self.als_iter_num = als_iter_num
#         self.mask_ratio = mask_ratio
#         self.tensor_decomposition_setting = {
#             "cp_rank": cp_rank,
#             "als_iter_num": als_iter_num,
#             "mask_ratio": mask_ratio,
#         }
#         self.trade_off_param = trade_off_param
#         self.distribution_type = distribution_type
#         self.rng = np.random.RandomState(seed)
#         self.unique_sampling = unique_sampling
#         # Internal storage
#         self._param_names = None
#         self._category_maps = None  # Map from param name to list of categories
#         self._shape = None  # Shape of the tensor
#         self._evaluated_indices = []  # List of indices of evaluated points
#         self._tensor_eval = None  # Tensor of evaluations
#         self._tensor_eval_bool = None  # Mask tensor indicating evaluated points
#         self._maximize = None

#     def infer_relative_search_space(self, study, trial):
#         # We focus on categorical parameters
#         search_space = optuna.search_space.intersection_search_space(
#             study.get_trials(deepcopy=False)
#         )
#         # Filter only categorical distributions
#         categorical_search_space = {}
#         for name, distribution in search_space.items():
#             if isinstance(distribution, optuna.distributions.CategoricalDistribution):
#                 categorical_search_space[name] = distribution
#         return categorical_search_space

#     def sample_relative(self, study, trial, search_space):
#         if not search_space:
#             return {}

#         # Initialize internal structures if not already done
#         if self._param_names is None:
#             self._param_names = list(search_space.keys())
#             self._param_names.sort()  # Ensure consistent order
#             self._category_maps = {}
#             self._shape = []
#             for param_name in self._param_names:
#                 distribution = search_space[param_name]
#                 categories = distribution.choices
#                 self._category_maps[param_name] = categories
#                 self._shape.append(len(categories))
#             self._shape = tuple(self._shape)
#             # Initialize tensors
#             self._tensor_eval = np.full(self._shape, np.nan)
#             self._tensor_eval_bool = np.zeros(self._shape, dtype=bool)
#             self._evaluated_indices = []
#             self._maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE

#         # Build tensor from past trials
#         self._update_tensor(study)

#         # Perform CP decomposition and suggest next parameter set
#         mean_tensor, std_tensor = self._fit(
#             self._tensor_eval,
#             self._tensor_eval_bool,
#             self._evaluated_indices,
#             self.distribution_type,
#         )

#         # Suggest next indices based on UCB
#         next_indices = self._suggest_ucb_candidates(
#             mean_tensor=mean_tensor,
#             std_tensor=std_tensor,
#             trade_off_param=self.trade_off_param,
#             batch_size=1,  # We only need one sample here
#             maximize=self._maximize,
#         )

#         # Convert indices back to parameter values
#         next_index = next_indices[0]
#         params = {}
#         for i, param_name in enumerate(self._param_names):
#             category_index = next_index[i]
#             category = self._category_maps[param_name][category_index]
#             params[param_name] = category

#         return params

#     def sample_independent(self, study, trial, param_name, param_distribution):
#         # Fallback to random sampling
#         return optuna.samplers.RandomSampler().sample_independent(
#             study, trial, param_name, param_distribution
#         )

#     def _update_tensor(self, study):
#         # Go through all completed trials and update the tensor
#         trials = study.get_trials(deepcopy=False)
#         for trial in trials:
#             if trial.state != optuna.trial.TrialState.COMPLETE:
#                 continue
#             # Get parameter indices
#             index = []
#             for param_name in self._param_names:
#                 if param_name not in trial.params:
#                     break  # Skip trials that don't have all parameters
#                 category = trial.params[param_name]
#                 try:
#                     category_index = self._category_maps[param_name].index(category)
#                 except ValueError:
#                     break  # Skip trials with unknown category
#                 index.append(category_index)
#             else:
#                 index = tuple(index)
#                 if index not in self._evaluated_indices:
#                     value = trial.value
#                     self._tensor_eval[index] = value
#                     self._tensor_eval_bool[index] = True
#                     self._evaluated_indices.append(index)

#     def _fit(
#         self,
#         tensor_eval: np.ndarray,
#         tensor_eval_bool: np.ndarray,
#         all_evaluated_indices: list[tuple[int, ...]],
#         distribution_type: str,
#     ) -> tuple[np.ndarray, np.ndarray]:
#         """
#         Perform CP decomposition and return the mean and variance tensors.
#         """
#         div = int(1 / self.mask_ratio)
#         mask_split_list = self._split_list_equally(all_evaluated_indices, div)

#         tensors_list = []

#         eval_min = np.nanmin(tensor_eval[np.isfinite(tensor_eval)]) # Avoid NaN
#         eval_max = np.nanmax(tensor_eval[np.isfinite(tensor_eval)]) # Avoid NaN
#         eval_mean = np.nanmean(tensor_eval[np.isfinite(tensor_eval)])
#         eval_std = np.nanstd(tensor_eval[np.isfinite(tensor_eval)])

#         # Generate initial tensor with random values
#         init_tensor_eval = self._generate_random_array(
#             low=eval_min, 
#             high=eval_max, 
#             shape=tensor_eval.shape,
#             distribution_type=distribution_type,
#             mean=eval_mean,
#             std_dev=eval_std,
#         )

#         init_tensor_eval[tensor_eval_bool == True] = tensor_eval[tensor_eval_bool == True]

#         for mask_list in mask_split_list:
#             mask_tensor = np.ones_like(tensor_eval_bool)
#             for mask_index in mask_list:
#                 mask_tensor[mask_index] = False

#             # print()
#             # print(init_tensor_eval)
#             # print()

#             # Perform CP decomposition
#             cp_tensor = parafac(
#                 init_tensor_eval,
#                 rank=self.cp_rank,
#                 mask=mask_tensor,
#                 n_iter_max=self.als_iter_num,
#                 init="random",
#                 random_state=self.rng,
#             )

#             # Convert the CP decomposition back to a tensor
#             reconstructed_tensor = cp_to_tensor(cp_tensor)

#             # Append the reconstructed tensor to the list for later processing
#             tensors_list.append(reconstructed_tensor)

#         # Calculate mean and variance tensors
#         tensors_stack = np.stack(tensors_list)
#         mean_tensor = np.mean(tensors_stack, axis=0)
#         std_tensor = np.std(tensors_stack, axis=0)

#         # Replace the mean and variance of known points with the original values and zeros
#         mean_tensor[tensor_eval_bool == True] = tensor_eval[tensor_eval_bool == True]
#         std_tensor[tensor_eval_bool == True] = 0

#         return mean_tensor, std_tensor

#     def _suggest_ucb_candidates(
#         self,
#         mean_tensor: np.ndarray,
#         std_tensor: np.ndarray,
#         trade_off_param: float,
#         batch_size: int,
#         maximize: bool,
#     ) -> list[tuple[int, ...]]:
#         """
#         Suggest candidate points based on UCB values, selecting the top batch_size points.

#         Returns:
#         - indices: list of tuples, the indices of the top batch_size points based on UCB.
#         """

#         def _ucb(mean_tensor, std_tensor, trade_off_param, maximize=True) -> np.ndarray:
#             mean_tensor = mean_tensor if maximize else -mean_tensor
#             ucb_values = mean_tensor + trade_off_param * std_tensor
#             return ucb_values

#         # Calculate the UCB values using the internal function
#         ucb_values = _ucb(mean_tensor, std_tensor, trade_off_param, maximize)

#         if self.unique_sampling:
#             # Mask out already evaluated points
#             ucb_values[self._tensor_eval_bool == True] = -np.inf

#         # Flatten the tensor and get the indices of the top UCB values
#         flat_indices = np.argsort(ucb_values.flatten())[::-1]  # Sort in descending order

#         top_indices = np.unravel_index(flat_indices[:batch_size], ucb_values.shape)
#         top_indices = list(zip(*top_indices))


#         self.ucb_temp = []

#         for index in top_indices:
#             logging.info(
#                 f"UCB value at {index}: {ucb_values[index]}, Mean: {mean_tensor[index]}, Std: {std_tensor[index]}"
#             )

#         return top_indices

#     def _split_list_equally(
#         self, input_list: list[tuple[int, ...]], div: int
#     ) -> list[list[tuple[int, ...]]]:
#         quotient, remainder = divmod(len(input_list), div)
#         result = []
#         start = 0
#         for i in range(div):
#             group_size = quotient + (1 if i < remainder else 0)
#             result.append(input_list[start : start + group_size])
#             start += group_size
#         return result

#     def _generate_random_array(
#         self,
#         low: float,
#         high: float,
#         shape: tuple[int, ...],
#         distribution_type: str = "uniform",
#         mean: float = 0,
#         std_dev: float = 1,
#     ) -> np.ndarray:
#         """
#         Generate an array of random numbers with specified bounds and distribution type.
#         Adds small noise if the evaluated entries are identical to avoid singular matrix errors.
#         """
#         # Handle case where all entries are identical by adding small random noise
#         if low == high:
#             low = low - 1e-6
#             high = high + 1e-6
#             std_dev = std_dev + 1e-6

#         if distribution_type == "uniform":
#             # Generate uniform random numbers
#             return self.rng.uniform(low, high, shape)
#         elif distribution_type == "normal":
#             # Generate normal random numbers and clip them to the specified bounds
#             normal_random = self.rng.normal(mean, std_dev, shape)
#             return np.clip(normal_random, low, high)
#         else:
#             raise ValueError("distribution_type must be either 'uniform' or 'normal'.")


# class NNParafacSampler(BaseSampler):
#     def __init__(
#         self,
#         cp_rank: int = 3,
#         als_iter_num: int = 10,
#         mask_ratio: float = 0.2,
#         trade_off_param: float = 1.0,
#         distribution_type: Literal["uniform", "normal"] = "uniform",
#         seed: int = None,
#         unique_sampling: bool = False,
#     ):
#         """
#         Initialize the ParafacSampler with the necessary settings.

#         Parameters:
#         - cp_rank: int, the rank of the CP decomposition.
#         - als_iter_num: int, the number of ALS iterations to perform during decomposition.
#         - mask_ratio: float, the ratio to control the number of masks for CP decomposition.
#         - trade_off_param: float, the trade-off parameter between exploration and exploitation.
#         - distribution_type: str, "uniform" or "normal" to choose the distribution type.
#         - seed: int, random seed.
#         """
#         self.cp_rank = cp_rank
#         self.als_iter_num = als_iter_num
#         self.mask_ratio = mask_ratio
#         self.trade_off_param = trade_off_param
#         self.distribution_type = distribution_type
#         self.rng = np.random.RandomState(seed)
#         self.unique_sampling = unique_sampling
#         # Internal storage
#         self._param_names = None
#         self._category_maps = None  # Map from param name to list of categories
#         self._shape = None  # Shape of the tensor
#         self._evaluated_indices = []  # List of indices of evaluated points
#         self._tensor_eval = None  # Tensor of evaluations
#         self._tensor_eval_bool = None  # Mask tensor indicating evaluated points
#         self._maximize = None

#     def infer_relative_search_space(self, study, trial):
#         # We focus on categorical parameters
#         search_space = optuna.search_space.intersection_search_space(
#             study.get_trials(deepcopy=False)
#         )
#         # Filter only categorical distributions
#         categorical_search_space = {}
#         for name, distribution in search_space.items():
#             if isinstance(distribution, optuna.distributions.CategoricalDistribution):
#                 categorical_search_space[name] = distribution
#         return categorical_search_space

#     def sample_relative(self, study, trial, search_space):
#         if not search_space:
#             return {}

#         # Initialize internal structures if not already done
#         if self._param_names is None:
#             self._param_names = list(search_space.keys())
#             self._param_names.sort()  # Ensure consistent order
#             self._category_maps = {}
#             self._shape = []
#             for param_name in self._param_names:
#                 distribution = search_space[param_name]
#                 categories = distribution.choices
#                 self._category_maps[param_name] = categories
#                 self._shape.append(len(categories))
#             self._shape = tuple(self._shape)
#             # Initialize tensors
#             self._tensor_eval = np.full(self._shape, np.nan)
#             self._tensor_eval_bool = np.zeros(self._shape, dtype=bool)
#             self._evaluated_indices = []
#             self._maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE

#         # Build tensor from past trials
#         self._update_tensor(study)

#         # Perform CP decomposition and suggest next parameter set
#         mean_tensor, std_tensor = self._fit(
#             self._tensor_eval,
#             self._tensor_eval_bool,
#             self._evaluated_indices,
#             self.distribution_type,
#         )

#         # Suggest next indices based on UCB
#         next_indices = self._suggest_ucb_candidates(
#             mean_tensor=mean_tensor,
#             std_tensor=std_tensor,
#             trade_off_param=self.trade_off_param,
#             batch_size=1,  # We only need one sample here
#             maximize=self._maximize,
#         )

#         # Convert indices back to parameter values
#         next_index = next_indices[0]
#         params = {}
#         for i, param_name in enumerate(self._param_names):
#             category_index = next_index[i]
#             category = self._category_maps[param_name][category_index]
#             params[param_name] = category

#         return params

#     def sample_independent(self, study, trial, param_name, param_distribution):
#         # Fallback to random sampling
#         return optuna.samplers.RandomSampler().sample_independent(
#             study, trial, param_name, param_distribution
#         )

#     def _update_tensor(self, study):
#         # Go through all completed trials and update the tensor
#         trials = study.get_trials(deepcopy=False)
#         for trial in trials:
#             if trial.state != optuna.trial.TrialState.COMPLETE:
#                 continue
#             # Get parameter indices
#             index = []
#             for param_name in self._param_names:
#                 if param_name not in trial.params:
#                     break  # Skip trials that don't have all parameters
#                 category = trial.params[param_name]
#                 try:
#                     category_index = self._category_maps[param_name].index(category)
#                 except ValueError:
#                     break  # Skip trials with unknown category
#                 index.append(category_index)
#             else:
#                 index = tuple(index)
#                 if index not in self._evaluated_indices:
#                     value = trial.value
#                     self._tensor_eval[index] = value
#                     self._tensor_eval_bool[index] = True
#                     self._evaluated_indices.append(index)

#     def _fit(
#         self,
#         tensor_eval: np.ndarray,
#         tensor_eval_bool: np.ndarray,
#         all_evaluated_indices: list[tuple[int, ...]],
#         distribution_type: str,
#     ) -> tuple[np.ndarray, np.ndarray]:
#         """
#         Perform CP decomposition and return the mean and variance tensors.
#         """
#         div = int(1 / self.mask_ratio)
#         mask_split_list = self._split_list_equally(all_evaluated_indices, div)

#         tensors_list = []

#         eval_min = np.nanmin(tensor_eval[np.isfinite(tensor_eval)]) # Avoid NaN
#         eval_max = np.nanmax(tensor_eval[np.isfinite(tensor_eval)]) # Avoid NaN
#         eval_mean = np.nanmean(tensor_eval[np.isfinite(tensor_eval)])
#         eval_std = np.nanstd(tensor_eval[np.isfinite(tensor_eval)])

#         # Generate initial tensor with random values
#         init_tensor_eval = self._generate_random_array(
#             low=eval_min, 
#             high=eval_max, 
#             shape=tensor_eval.shape,
#             distribution_type=distribution_type,
#             mean=eval_mean,
#             std_dev=eval_std,
#         )

#         init_tensor_eval[tensor_eval_bool == True] = tensor_eval[tensor_eval_bool == True]

#         for mask_list in mask_split_list:
#             mask_tensor = np.ones_like(tensor_eval_bool)
#             for mask_index in mask_list:
#                 mask_tensor[mask_index] = False

#             # Perform CP decomposition
#             cp_tensor = non_negative_parafac(
#                 init_tensor_eval,
#                 rank=self.cp_rank,
#                 mask=mask_tensor,
#                 n_iter_max=self.als_iter_num,
#                 init="random",
#                 random_state=self.rng,
#             )

#             # Convert the CP decomposition back to a tensor
#             reconstructed_tensor = cp_to_tensor(cp_tensor)

#             # Append the reconstructed tensor to the list for later processing
#             tensors_list.append(reconstructed_tensor)

#         # Calculate mean and variance tensors
#         tensors_stack = np.stack(tensors_list)
#         mean_tensor = np.mean(tensors_stack, axis=0)
#         std_tensor = np.std(tensors_stack, axis=0)

#         # Replace the mean and variance of known points with the original values and zeros
#         mean_tensor[tensor_eval_bool == True] = tensor_eval[tensor_eval_bool == True]
#         std_tensor[tensor_eval_bool == True] = 0

#         return mean_tensor, std_tensor

#     def _suggest_ucb_candidates(
#         self,
#         mean_tensor: np.ndarray,
#         std_tensor: np.ndarray,
#         trade_off_param: float,
#         batch_size: int,
#         maximize: bool,
#     ) -> list[tuple[int, ...]]:
#         """
#         Suggest candidate points based on UCB values, selecting the top batch_size points.

#         Returns:
#         - indices: list of tuples, the indices of the top batch_size points based on UCB.
#         """

#         def _ucb(mean_tensor, std_tensor, trade_off_param, maximize=True) -> np.ndarray:
#             mean_tensor = mean_tensor if maximize else -mean_tensor
#             ucb_values = mean_tensor + trade_off_param * std_tensor
#             return ucb_values

#         # Calculate the UCB values using the internal function
#         ucb_values = _ucb(mean_tensor, std_tensor, trade_off_param, maximize)

#         if self.unique_sampling:
#             # Mask out already evaluated points
#             ucb_values[self._tensor_eval_bool == True] = -np.inf

#         # Flatten the tensor and get the indices of the top UCB values
#         flat_indices = np.argsort(ucb_values.flatten())[::-1]  # Sort in descending order

#         top_indices = np.unravel_index(flat_indices[:batch_size], ucb_values.shape)
#         top_indices = list(zip(*top_indices))

#         for index in top_indices:
#             _logger.debug(
#                 f"UCB value at {index}: {ucb_values[index]}, Mean: {mean_tensor[index]}, Std: {std_tensor[index]}"
#             )

#         return top_indices

#     def _split_list_equally(
#         self, input_list: list[tuple[int, ...]], div: int
#     ) -> list[list[tuple[int, ...]]]:
#         quotient, remainder = divmod(len(input_list), div)
#         result = []
#         start = 0
#         for i in range(div):
#             group_size = quotient + (1 if i < remainder else 0)
#             result.append(input_list[start : start + group_size])
#             start += group_size
#         return result

#     def _generate_random_array(
#         self,
#         low: float,
#         high: float,
#         shape: tuple[int, ...],
#         distribution_type: str = "uniform",
#         mean: float = 0,
#         std_dev: float = 1,
#     ) -> np.ndarray:
#         """
#         Generate an array of random numbers with specified bounds and distribution type.
#         Adds small noise if the evaluated entries are identical to avoid singular matrix errors.
#         """
#         # Handle case where all entries are identical by adding small random noise
#         if low == high:
#             low = low - 1e-6
#             high = high + 1e-6
#             std_dev = std_dev + 1e-6

#         if distribution_type == "uniform":
#             # Generate uniform random numbers
#             return self.rng.uniform(low, high, shape)
#         elif distribution_type == "normal":
#             # Generate normal random numbers and clip them to the specified bounds
#             normal_random = self.rng.normal(mean, std_dev, shape)
#             return np.clip(normal_random, low, high)
#         else:
#             raise ValueError("distribution_type must be either 'uniform' or 'normal'.")