from typing import List

import numpy as np
from transformers import XLNetTokenizer


def parse_humicroedit(
    dataset, expression_separator: str = "{", expression_end: str = "}"
) -> List[str]:
    """
    Parse the humicrocredit dataset in an appropriate format.
    - token separating numbers from text: {
    - oken separating text items: }

    Args:
        dataset: The respective chunkk of the humicroedit dataset loaded via Huggingface.

    Raises:
        ValueError: If the joke cant be extracted uniquely

    Returns:
        _description_
    """

    lines = []
    for sample in dataset:
        prop = "[funny]" + str(round(float(sample["meanGrade"]), 1))
        text = sample["original"]
        if text.count("<") > 1 or text.count("/>") > 1:
            raise ValueError(text)
        if "{" in text or "}" in text:
            print(text)
        text = text.replace("<", "START ").replace("/>", " END")

        line = prop + expression_separator + sample["edit"] + expression_end + text
        lines.append(line)
    return lines


def compute_topk(predictions: np.array) -> List[float]:
    """
    Computes the topk accuracy of a boolean np array

    Args:
        predictions: boolean np.array of shape batch_size x k with correctness of each
            prediction

    Returns:
        List of floats denoting the top-k accuracies
    """

    topk = [np.mean(predictions[:, 0])]
    for k in range(1, predictions.shape[1]):
        topk.append(topk[-1] + np.mean(predictions[:, k]))
    return topk
