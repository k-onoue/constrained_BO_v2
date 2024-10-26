import configparser
import sys

# Load configuration
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
LOG_DIR = config["paths"]["logs_dir"]
DB_DIR = config["paths"]["dbs_dir"]
sys.path.append(PROJECT_DIR)

from src.objectives.warcraft import WarcraftObjective
from src.samplers.parafac import ParafacSampler
from src.utils_experiments import set_logger, parse_experiment_path

__all__ = [
    "WarcraftObjective",
    "ParafacSampler",
    "set_logger",
    "parse_experiment_path",
]
