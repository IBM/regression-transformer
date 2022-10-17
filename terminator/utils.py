import logging
import os
import subprocess as sp
import sys
from typing import List

import numpy as np
import psutil
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_gpu_memory():
    if not cuda():
        return 0, 0, 0
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

    tot_m, used_m, free_m = map(int, os.popen("free -t -m").readlines()[-1].split()[1:])
    return memory_free_values, used_m, tot_m


def get_cpu_memory():
    mem = psutil.virtual_memory()
    return mem.total / 1000**3, mem.percent, psutil.cpu_percent()


def get_process_mmeory():
    process = psutil.Process(os.getpid())
    return process.memory_percent()


def get_device():
    return torch.device("cuda" if cuda() else "cpu")


def cuda():
    return torch.cuda.is_available()


def get_latest_checkpoint(model_path: str, must_contain: str = "best") -> str:
    """
    Given a path to the model folder it searches the latest saved checkpoint
    and returns the path to it.
    Args:
        model_path (str): Path to model folder. Has to contain folders called
            'checkpoint-best-STEP' and 'checkpoint-latest-STEP' where STEP is
            a positive integer.
        must_contain (str, optional): Subselect checkpoints that contain a
            certain query. Defaults to 'best'.
    Returns:
        str: Path to latest checkpoint
    """

    # Finding checkpoints
    checkpoints = [f for f in os.listdir(model_path) if f.startswith("checkpoint")]
    if must_contain is not None:
        checkpoints = list(filter(lambda x: must_contain in x, checkpoints))

    if len(checkpoints) == 0:
        logger.warning(
            f"No checkpoints found that contain {must_contain} in {model_path}."
        )
        # Relax criteria and retry
        next_try = "checkpoint" if must_contain != "checkpoint" else ""
        return get_latest_checkpoint(model_path, must_contain=next_try)

    # Sorting
    try:
        idx = np.argsort([int(c.split("-")[-1]) for c in checkpoints])[-1]
    except ValueError:
        raise ValueError(f"Checkpoints dont seem to follow format: {checkpoints}.")

    return os.path.join(model_path, checkpoints[idx])


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog("rdApp.error")


def find_safe_path(path: str) -> str:
    """Method to find a safe path that does not exist yet.
    Args:
        path (str): Desired path.
    Returns:
        str: Non existing path.
    """
    safe_path = path
    c = 0
    while os.path.exists(safe_path):
        c += 1
        safe_path = ".".join(
            [
                s if i != path.count(".") - 1 else f"{s}_v{c}"
                for i, s in enumerate(path.split("."))
            ]
        )
    return safe_path


def get_equispaced_ranges(
    data_path: str, properties: List[str], n: int = 10, precisions: List[int] = [2]
) -> List[List[float]]:
    """
    Given a path to a data file it returns the ranges of the properties.
    Args:
        data_path : Path to data file.
        properties: List of properties to consider.
        n: number of points per property (will be equally spaced).
        precisions: number of decimal places to round to (one per property).
    Returns:
        List of ranges for each property.
    """
    with open(data_path, "r") as f:
        data = f.readlines()

    ranges = []

    for prop, pre in zip(properties, precisions):

        values = [float(line.split(prop)[-1].split("|")[0]) for line in data]
        _range = []
        for x in np.linspace(np.min(values), np.max(values), n):
            if pre == 1:
                _range.append(f"{x:.1f}")
            elif pre == 2:
                _range.append(f"{x:.2f}")
            elif pre == 3:
                _range.append(f"{x:.3f}")
            elif pre == 4:
                _range.append(f"{x:.4f}")
        ranges.append(_range)
    return ranges
