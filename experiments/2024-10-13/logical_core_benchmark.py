import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np
import logging
import time
import random
import argparse
from _src import LOG_DIR
from _src import set_logger


# argparseの設定
parser = argparse.ArgumentParser(description='Run Parafac decomposition benchmark with specified logical cores.')
parser.add_argument('--logical_cores', type=int, required=True, help='Number of logical cores to use for the experiment')
args = parser.parse_args()

logical_cores = args.logical_cores

# ログファイルの設定
logfile_name = f"benchmark_log_cores_{logical_cores}.log"
set_logger(logfile_name, LOG_DIR)

# テンソルの次元リスト
shapes = [(7,) * 4, (7,) * 9, (7,) * 11]
num_trials = 5

# パラファック分解（CPRank2）の実行と再構成
def perform_parafac_and_reconstruct(tensor, rank=2):
    start_time = time.time()
    
    # Parafac 分解
    factors = parafac(tensor, rank=rank)
    
    # 元のテンソルに戻す
    reconstructed_tensor = tl.kruskal_to_tensor(factors)
    
    # 計算時間
    elapsed_time = time.time() - start_time
    return elapsed_time

# それぞれのテンソル次元での計算
for shape in shapes:
    times = []
    
    logging.info(f"Running Parafac decomposition for tensor shape {shape} with {logical_cores} logical cores")
    
    for trial in range(num_trials):
        # シード値を設定
        np.random.seed(trial)
        random.seed(trial)
        
        # ランダムなテンソルを生成
        tensor = np.random.rand(*shape)
        
        # 分解と再構成の実行
        elapsed_time = perform_parafac_and_reconstruct(tensor)
        times.append(elapsed_time)
        
        logging.info(f"Trial {trial+1}: {elapsed_time:.6f} seconds")
    
    # 平均と分散を計算
    mean_time = np.mean(times)
    variance_time = np.var(times)
    
    logging.info(f"Shape {shape} - Average Time: {mean_time:.6f} seconds, Variance: {variance_time:.6f} seconds")

logging.info(f"Benchmark completed. Log saved to {logfile_name}")