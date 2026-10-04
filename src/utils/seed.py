"""
Reproducibility Utilities
Ensures deterministic random seeds across standard Python libraries, NumPy, and PyTorch.
"""
import os
import random
import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """
    Set random seeds across Python random, NumPy, OS, and PyTorch (if installed).

    Args:
        seed (int): The integer seed value to enforce.
        deterministic (bool): If True, configures CuDNN and backend operations to be deterministic.

    Returns:
        int: The seed value that was applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
