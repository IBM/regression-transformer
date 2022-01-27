import logging
import os
import sys
import transformers
import numpy as np
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


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
        logger.warning(f"No checkpoints found that contain {must_contain} in {model_path}.")
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
        safe_path = ".".join([s if i != path.count(".") - 1 else f"{s}_v{c}" for i, s in enumerate(path.split("."))])
    return safe_path
