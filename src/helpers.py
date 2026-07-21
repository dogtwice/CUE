import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
import os
import warnings
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from . import config


def npy(t, to_cpu=True):
    """
    Convert a tensor to a numpy array.

    :param t: Input tensor
    :type t: th.Tensor
    :param to_cpu: Call the .cpu() method on `t`?
    :type to_cpu: bool
    :return: Numpy array
    :rtype: np.ndarray
    """
    if isinstance(t, (list, tuple)):
        # We got a list. Convert each element to numpy
        return [npy(ti) for ti in t]
    elif isinstance(t, dict):
        # We got a dict. Convert each value to numpy
        return {k: npy(v) for k, v in t.items()}
    # Assuming t is a tensor.
    if to_cpu:
        arr = t.cpu().detach().numpy()
    else:
        arr = t.detach().numpy()
    # 如果是标量（0维数组），转换为Python标量以便pandas处理
    if arr.ndim == 0:
        return arr.item()
    return arr


def dict_means(dicts):
    """
    Compute the mean value of keys in a list of dicts

    :param dicts: Input dicts
    :type dicts: List[dict]
    :return: Mean values
    :rtype: dict
    """
    return pd.DataFrame(dicts).mean(axis=0).to_dict()


def add_prefix(dct, prefix, sep="/"):
    """
    Add a prefix to all keys in `dct`.

    :param dct: Input dict
    :type dct: dict
    :param prefix: Prefix
    :type prefix: str
    :param sep: Separator between prefix and key
    :type sep: str
    :return: Dict with prefix prepended to all keys
    :rtype: dict
    """
    return {prefix + sep + key: value for key, value in dct.items()}


def ordered_cmat(labels, pred):
    """
    Compute the confusion matrix and accuracy corresponding to the best cluster-to-class assignment.

    :param labels: Label array
    :type labels: np.array
    :param pred: Predictions array
    :type pred: np.array
    :return: Accuracy and confusion matrix
    :rtype: Tuple[float, np.array]
    """
    cmat = confusion_matrix(labels, pred)
    ri, ci = linear_sum_assignment(-cmat)
    ordered = cmat[np.ix_(ri, ci)]
    acc = np.sum(np.diag(ordered))/np.sum(ordered)
    return acc, ordered


def align_labels(labels, pred):
    """
    Align predicted cluster labels to ground truth labels using the Hungarian algorithm.
    
    :param labels: Ground truth labels
    :type labels: np.array
    :param pred: Predicted cluster labels
    :type pred: np.array
    :return: Aligned predictions
    :rtype: np.array
    """
    labels = np.asarray(labels)
    pred = np.asarray(pred)
    
    cmat = confusion_matrix(labels, pred)
    ri, ci = linear_sum_assignment(-cmat)
    
    # 创建映射：cluster_id -> true_class_id
    mapping = {cluster_id: true_class_id for true_class_id, cluster_id in zip(ri, ci)}
    
    # 应用映射
    aligned_pred = np.array([mapping.get(p, p) for p in pred])
    return aligned_pred


def get_save_dir(experiment_name, identifier, run, dataset_name=None, artifact_type="main"):
    """
    Get the save dir for an experiment

    :param experiment_name: Name of the config
    :type experiment_name: str
    :param identifier: 8-character unique identifier for the current experiment
    :type identifier: str
    :param run: Current training run
    :type run: int
    :return: Path to save dir
    :rtype: pathlib.Path
    """
    from . import artifacts

    dataset_name = dataset_name or experiment_name
    experiment_dir = artifacts.get_training_experiment_dir(
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        identifier=identifier,
        category=artifact_type,
    )
    return artifacts.get_run_dir(experiment_dir, run)


def he_init_weights(module):
    """
    Initialize network weights using the Xavier Uniform initialization strategy.
    
    Xavier Uniform is recommended for networks with tanh/sigmoid activations,
    and also works well with ReLU in graph neural networks.

    :param module: Network module
    :type module: nn.Module
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def set_seed(seed, random_mode=False):
    """
    Set random seed for reproducibility.

    :param seed: Random seed
    :type seed: int
    :param random_mode: If True, skips seed setting to allow full randomness
    :type random_mode: bool
    """
    if random_mode:
        # If random mode is enabled, we still enable benchmark for speed
        # and disable deterministic behavior
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for some PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    warnings.filterwarnings(
        "ignore",
        message=r".*CUBLAS_WORKSPACE_CONFIG.*",
    )


def make_torch_generator(seed, random_mode=False, offset=0):
    """
    Build a deterministic torch.Generator for DataLoader shuffling when
    fixed-seed mode is enabled. Returns None in random mode.
    """
    if random_mode:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


def make_worker_init_fn(seed, random_mode=False):
    """
    Build a worker_init_fn that seeds Python, NumPy and Torch RNGs for each
    DataLoader worker in fixed-seed mode. Returns None in random mode.
    """
    if random_mode:
        return None

    return SeedWorkerInitFn(seed)


class SeedWorkerInitFn:
    """
    Pickle-safe DataLoader worker initializer for Windows multiprocessing.
    """

    def __init__(self, seed):
        self.seed = int(seed)

    def __call__(self, worker_id):
        worker_seed = (self.seed + int(worker_id)) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

