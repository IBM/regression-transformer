import re
from typing import Any, Dict, Optional

import numpy as np
import torch
import transformers
from torch import Tensor
from transformers.utils import logging

logger = logging.get_logger(__name__)


def get_trainer_dict(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to take out a subset of a dictionary with keys that are
    important for `CustomTrainer` but cant be passed down to `Trainer`.

    Args:
        dictionary (dict): Dict with keyword arguments for `CustomTrainer` constructor.

    Returns:
        dict: Dict with keyword arguments for `CustomTrainer` that cant be passed to
            childclass constructor (`Trainer`).
    """
    keys_to_keep = [
        "verbose_evaluation",
        "numerical",
        "d_model",
        "vocab_size",
        "vmax",
        "model_type",
        "mem_len",
        "training_logs",
        "train_config",
        "alternating_collator",
    ]
    keep_dict = {}
    for keep_key in keys_to_keep:
        for key, val in dictionary.items():
            if re.search(keep_key, key) is not None:
                keep_dict[key] = val
    return keep_dict


"""
All below code is taken from transformers==3.5.0 to remedy issues with tensor stacking.
NOTE: 3.4.0 introduces accumulation steps in evaluation, but only 3.5.0 allows the
Trainer to handle dynamic sequence lengths.
"""


def nested_new_like(arrays, num_samples, padding_index=-100):
    """Create the same nested structure as `arrays` with a first dimension always at `num_samples`."""
    if isinstance(arrays, (list, tuple)):
        return type(arrays)(nested_new_like(x, num_samples) for x in arrays)
    return np.full_like(arrays, padding_index, shape=(num_samples, *arrays.shape[1:]))


def nested_truncate(tensors, limit):
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    return tensors[:limit]


def nested_expand_like(arrays, new_seq_length, padding_index=-100):
    """Expand the `arrays` so that the second dimension grows to `new_seq_length`.
    Uses `padding_index` for padding."""
    if isinstance(arrays, (list, tuple)):
        return type(arrays)(
            nested_expand_like(x, new_seq_length, padding_index=padding_index)
            for x in arrays
        )

    result = np.full_like(
        arrays,
        padding_index,
        shape=(arrays.shape[0], new_seq_length) + arrays.shape[2:],
    )
    result[:, : arrays.shape[1]] = arrays
    return result


def _get_first_shape(arrays):
    """Return the shape of the first array found in the nested struct `arrays`."""
    if isinstance(arrays, (list, tuple)):
        return _get_first_shape(arrays[0])
    return arrays.shape


class DistributedTensorGatherer:
    """
    A class responsible for properly gathering tensors (or nested list/tuple of tensors) on the CPU
    by chunks.
    If our dataset has 16 samples with a batch size of 2 on 3 processes and we gather then transfer on
    CPU at every step, our sampler will generate the following indices:
        :obj:`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]`
    to get something of size a multiple of 3 (so that each process gets the same dataset length). Then
    process 0, 1 and 2 will be responsible of making predictions for the following samples:
        - P0: :obj:`[0, 1, 2, 3, 4, 5]`
        - P1: :obj:`[6, 7, 8, 9, 10, 11]`
        - P2: :obj:`[12, 13, 14, 15, 0, 1]`
    The first batch treated on each process will be
        - P0: :obj:`[0, 1]`
        - P1: :obj:`[6, 7]`
        - P2: :obj:`[12, 13]`
    So if we gather at the end of the first batch, we will get a tensor (nested list/tuple of tensor)
    corresponding to the following indices:
        :obj:`[0, 1, 6, 7, 12, 13]`
    If we directly concatenate our results without taking any precautions, the user will then get
    the predictions for the indices in this order at the end of the prediction loop:
        :obj:`[0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1]`
    For some reason, that's not going to roll their boat. This class is there to solve that problem.
    Args:
        world_size (:obj:`int`):
            The number of processes used in the distributed training.
        num_samples (:obj:`int`):
            The number of samples in our dataset.
        make_multiple_of (:obj:`int`, `optional`):
            If passed, the class assumes the datasets passed to each process are made to be a multiple of this argument
            (by adding samples).
    """

    def __init__(
        self, world_size, num_samples, make_multiple_of=None, padding_index=-100
    ):
        self.world_size = world_size
        self.num_samples = num_samples
        total_size = (
            world_size if make_multiple_of is None else world_size * make_multiple_of
        )
        self.total_samples = int(np.ceil(num_samples / total_size)) * total_size
        self.process_length = self.total_samples // world_size
        self._storage = None
        self._offsets = None
        self.padding_index = padding_index

    def add_arrays(self, arrays):
        """
        Add :obj:`arrays` to the internal storage, Will initialize the storage to the full size at the first arrays
        passed so that if we're bound to get an OOM, it happens at the beginning.
        """
        if arrays is None:
            return
        if self._storage is None:
            self._storage = nested_new_like(
                arrays, self.total_samples, padding_index=self.padding_index
            )
            self._offsets = list(range(0, self.total_samples, self.process_length))
        else:
            storage_shape = _get_first_shape(self._storage)
            arrays_shape = _get_first_shape(arrays)
            if len(storage_shape) > 1 and storage_shape[1] < arrays_shape[1]:
                # If we get new arrays that are too big too fit, we expand the shape fo the storage
                self._storage = nested_expand_like(
                    self._storage, arrays_shape[1], padding_index=self.padding_index
                )
        slice_len = self._nested_set_tensors(self._storage, arrays)
        for i in range(self.world_size):
            self._offsets[i] += slice_len

    def _nested_set_tensors(self, storage, arrays):
        if isinstance(arrays, (list, tuple)):
            for x, y in zip(storage, arrays):
                slice_len = self._nested_set_tensors(x, y)
            return slice_len
        assert (
            arrays.shape[0] % self.world_size == 0
        ), f"Arrays passed should all have a first dimension multiple of {self.world_size}, found {arrays.shape[0]}."

        slice_len = arrays.shape[0] // self.world_size
        for i in range(self.world_size):
            if len(arrays.shape) == 1:
                storage[self._offsets[i] : self._offsets[i] + slice_len] = arrays[
                    i * slice_len : (i + 1) * slice_len
                ]
            else:
                storage[
                    self._offsets[i] : self._offsets[i] + slice_len, : arrays.shape[1]
                ] = arrays[i * slice_len : (i + 1) * slice_len]
        return slice_len

    def finalize(self):
        """
        Return the properly gathered arrays and truncate to the number of samples (since the sampler added some extras
        to get each process a dataset of the same length).
        """
        if self._storage is None:
            return
        if self._offsets[0] != self.process_length:
            logger.warn(
                "Not all data has been set. Are you sure you passed all values?"
            )
        return nested_truncate(self._storage, self.num_samples)


def torch_pad_and_concatenate(
    tensor1: Tensor, tensor2: Tensor, padding_index: int = -100
) -> Tensor:
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (
        tensor1.shape[0] + tensor2.shape[0],
        max(tensor1.shape[1], tensor2.shape[1]),
    ) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result.detach()


def numpy_pad_and_concatenate(
    array1: np.array, array2: np.array, padding_index: str = -100
) -> np.array:
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), dim=0)

    # Let's figure out the new shape
    new_shape = (
        array1.shape[0] + array2.shape[0],
        max(array1.shape[1], array2.shape[1]),
    ) + array1.shape[2:]

    # Now let's fill the result tensor
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(
            nested_concat(t, n, padding_index=padding_index)
            for t, n in zip(tensors, new_tensors)
        )
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(
            tensors, new_tensors, padding_index=padding_index
        )
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(
            tensors, new_tensors, padding_index=padding_index
        )
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")


def distributed_concat(
    tensor: "torch.Tensor", num_total_examples: Optional[int] = None
) -> torch.Tensor:
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(
                distributed_concat(t, num_total_examples) for t in tensor
            )
        output_tensors = [
            tensor.clone() for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    return tensors.cpu().numpy()
