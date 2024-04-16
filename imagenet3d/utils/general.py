import logging
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            import transformers
            transformers.set_seed(seed)
        except ImportError:
            pass
        logging.info(f"Set random seed to {seed}")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_pkg_root():
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(root, ".."))
    return root


def get_project_root():
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(root, "..", ".."))
    return root


def get_abs_path(path):
    if not os.path.isabs(path):
        path = os.path.join(get_project_root(), path)
    return path


def setup_logging(save_path, log_name=None):
    save_path = get_abs_path(save_path)
    os.makedirs(save_path, exist_ok=True)
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_name is None:
        filename = os.path.join(save_path, f"log_{dt}.txt")
    else:
        filename = os.path.join(save_path, f"log_{log_name}_{dt}.txt")
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=filename,
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    return logging.getLogger("").handlers[0].baseFilename
