import tensorly as tl
from tensorly.decomposition import tensor_ring
import numpy as np
import logging
import time
import random
import argparse
from _src import LOG_DIR
from _src import set_logger
from sklearn.metrics import root_mean_squared_error

# argparseの設定
parser = argparse.ArgumentParser(description='Run Tensor Ring decomposition benchmark with specified logical cores and rank.')
parser.add_argument('--logical_cores', type=int, required=True, help='Number of logical cores to use for the experiment')
parser.add_argument('--rank', type=int, required=True, help='Index of rank to use from the rank_list')
args = parser.parse_args()

logical_cores = args.logical_cores
rank_index = args.rank

# ログファイルの設定
logfile_name = f"benchmark_log_cores_{logical_cores}_rank{rank_index}_tr"
set_logger(logfile_name, LOG_DIR)

# テンソルの次元リスト
shapes = [(7,) * 4, (7,) * 9, (7,) * 11]
num_trials = 5

# Rank list (rank_index is used to select the appropriate rank)
rank_list = [1, 2, 3, 4]
selected_rank = rank_list[rank_index]

# テンソルリング分解（TR）の実行と再構成
def perform_tr_and_reconstruct(tensor, rank=1):
    start_time = time.time()

    # Tensor Ring 分解
    factors = tensor_ring(tensor, rank=rank)
    
    # 元のテンソルに戻す
    reconstructed_tensor = tl.tr_to_tensor(factors)
    
    # 計算時間
    elapsed_time = time.time() - start_time
    
    # Reconstruction accuracy (Mean Squared Error)
    reconstruction_error = root_mean_squared_error(tensor.ravel(), reconstructed_tensor.ravel())
    
    return elapsed_time, reconstruction_error

# それぞれのテンソル次元での計算
for shape in shapes:
    times = []
    reconstruction_errors = []
    
    logging.info(f"Running Tensor Ring decomposition for tensor shape {shape} with {logical_cores} logical cores and rank {selected_rank}")
    
    for trial in range(num_trials):
        print(f"Trial {trial+1}")
        # シード値を設定
        np.random.seed(trial)
        random.seed(trial)
        
        # ランダムなテンソルを生成
        tensor = np.random.rand(*shape)
        
        # 分解と再構成の実行
        elapsed_time, reconstruction_error = perform_tr_and_reconstruct(tensor, rank=selected_rank)
        times.append(elapsed_time)
        reconstruction_errors.append(reconstruction_error)
        
        logging.info(f"Trial {trial+1}: {elapsed_time:.6f} seconds, Reconstruction Error: {reconstruction_error:.6f}")
    
    # 平均と分散を計算
    mean_time = np.mean(times)
    variance_time = np.var(times)
    mean_reconstruction_error = np.mean(reconstruction_errors)
    variance_reconstruction_error = np.var(reconstruction_errors)
    
    logging.info(f"Shape {shape} - Average Time: {mean_time:.6f} seconds, Variance: {variance_time:.6f} seconds")
    logging.info(f"Shape {shape} - Average Reconstruction Error: {mean_reconstruction_error:.6f}, Variance of Reconstruction Error: {variance_reconstruction_error:.6f}")

logging.info(f"Benchmark completed. Log saved to {logfile_name}")
