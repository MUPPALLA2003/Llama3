import torch
import os
import random
import numpy as np

def set_seed(seed,deterministic:bool=True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic: 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True,warn_only = True)

    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_torch_generator(seed:int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g