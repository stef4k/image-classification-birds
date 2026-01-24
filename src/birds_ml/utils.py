import os, random
import numpy as np

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path
