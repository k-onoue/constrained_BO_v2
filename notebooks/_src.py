import configparser
import sys

# Load configuration
config = configparser.ConfigParser()
config_path = "./../config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
LOG_DIR = config["paths"]["logs_dir"]
sys.path.append(PROJECT_DIR)

from src.utils_warcraft import (
    judge_location_validity,
    get_d_to,
    navigate_through_matrix,
    manhattan_distance,
    get_opposite,
    get_next_coordinate,
    judge_continuity,
)

__all__ = [
    "WarcraftObjectiveBoTorch",
    "get_opposite",
    "judge_continuity",
    "get_next_coordinate",
    "judge_location_validity",
    "get_d_to",
    "navigate_through_matrix",
    "manhattan_distance",
]
