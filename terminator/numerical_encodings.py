import numbers
import warnings
from math import cos, inf, sin
from typing import Dict, Optional

import torch
import torch.nn as nn
import transformers
from torch import Tensor

from .utils import get_device


def get_float_encoding(
    token: str, embedding_size: int, vmax: float = 1.0
) -> torch.Tensor:
    """Convert a token representing a float into a _fixed_ embedding vector.
    NOTE: This can be used for *any* range of numbers > 0.

    Args:
        token (str): A token representing a float. NOTE: Needs to follow notation
            _8_-1_ to represent 0.8 or _5_-2_ to represent 0.05.
        embedding_size (int): Size of the embedding.
        vmax (int, optional): Maximal value of float, defaults to 1. Normalizes
            values to be in the range ~ [-10, 10].
            NOTE: If remaining nn.embeddings in model use `max_norm`, this might result
            in large range discrepancies.

    Returns:
        torch.Tensor: Tensor of length embedding_size containing the embedding.
    """
    if embedding_size % 2 != 0:
        raise ValueError("Embedding size cant be odd.")

    vals = torch.zeros((embedding_size,))
    if len(token) == 1 or not (
        token.startswith("_") and token.endswith("_") and token.count("_") == 3
    ):
        return vals
    else:
        digit = int(token[1])
        order = int(token.split("_")[-2])
        val = digit * 10**order

    for i in range(0, embedding_size, 2):
        vals[i] = val / (i + 1)
        vals[i + 1] = -val / (i + 1)

    return vals / (vmax / 10)


def get_full_float_encoding(
    value: float, embedding_size: int, vmax: float = 1.0
) -> Tensor:
    """
    Convert a float value into a _fixed_ embedding vector.

    Args:
        value: The float value to be encoded.
        embedding_size: The size of the embedding.
        vmax: Maximal value the `value` variable can take. This normalizes values
            to be in the range ~ [-10, 10]. NOTE: If remaining nn.embeddings in
            model use `max_norm`, this might result in large range discrepancies.

    Returns:
        torch.Tensor of shape (embedding_size, ) containing the embedding.
    """
    if embedding_size % 2 != 0:
        raise ValueError(f"Embedding size {embedding_size} cant be odd.")
    integer = int(value)
    decimal = value - integer
    scalar = integer * 10**decimal
    embedding = torch.zeros((embedding_size,))
    for i in range(0, embedding_size, 2):
        embedding[i] = scalar / (i + 1)
        embedding[i + 1] = -scalar / (i + 1)
    return embedding


def get_int_encoding(token: str, embedding_size: int) -> torch.Tensor:
    """Convert a token representing an integer into a _fixed_ embedding vector.
    NOTE: This can be used only for positive integers - the generation of the
        encodings is *identical* to positional encodings.

    Args:
        token (str): A token representing an integer. NOTE: Needs to follow notation
            _8_2_ to represent 80 or _5_1_ to represent 5.
        embedding_size (int): Size of the embedding.

    Returns:
        torch.Tensor: Tensor of length embedding_size containing the embedding.
    """
    ed = embedding_size
    vals = torch.zeros((ed,))

    if len(token) == 1 or not (
        token.startswith("_") and token.endswith("_") and token.count("_") == 3
    ):
        return vals
    else:
        digit = int(token[1])
        order = int(token.split("_")[-2])
        val = digit * 10**order

        if order < 0:
            raise ValueError(
                f"Found float encoding in {token}. Pass positive ints only."
            )

    sine = lambda p, i: sin(p / (10000.0 ** (2 * i / ed)))
    cose = lambda p, i: cos(p / (10000.0 ** (2 * i / ed)))
    for i in range(0, ed, 2):
        vals[i] = sine(val, i)
        vals[i + 1] = cose(val, i)
    return vals


class FloatEncoding(nn.Embedding):
    """
    A nn.Embedding inspired class to generate fixed embedding vectors that represent
    numbers passed as tokens.
    NOTE: Tokens representing numbers need to follow notation _8_-1_ to represent 0.8.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        vocab: Dict,
        vmax: Optional[float] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Constructor for FloatEmbedding; sets up the fixed embedding matrix.

        Args:
            num_embeddings (int): size of the dictionary of embeddings.
            embedding_dim (int): the size of each embedding vector
            vocab (Dict): the language dictionary with tokens as keys and indexes as
                values. Length needs to match num_embeddings
            vmax (Optional[float]): Maximal value of float, defaults to None.

        Raises:
            ValueError: if num_embeddings does not match len(vocab).
            TypeError: if neither None nor a number is passed as vmax
            ValueError: if vmax is negative.
        """

        super(FloatEncoding, self).__init__(
            num_embeddings, embedding_dim, *args, **kwargs
        )

        if not len(vocab) == num_embeddings:
            raise ValueError(
                f"num_embeddings needs to match size of vocabulary ({num_embeddings}!={len(vocab)})"
            )
        if not (vmax is None or isinstance(3, numbers.Number)):
            raise TypeError(f"vmax needs to be a number or None, not {vmax}.")

        if vmax is None:
            # Infer the highest number in the dictionary (for normalization)
            test = lambda t: len(t) == 1 or not (
                t.startswith("_") and t.endswith("_") and t.count("_") == 3
            )
            vmax = max(
                [
                    -inf
                    if test(token)
                    else int(token[1]) * 10 ** int(token.split("_")[-2])
                    for token in vocab.keys()
                ]
            )
            warnings.warn(
                f"The inferred maximum float ({vmax}) is used for normalizing all float embeddings"
                " which might result in diminishing embeddings."
            )

        if vmax < 0:
            raise ValueError(f"Can not work only with negative numbers (vmax = {vmax})")

        weights = torch.zeros(num_embeddings, embedding_dim)
        for idx, (token, index) in enumerate(vocab.items()):
            assert (
                idx == index
            ), "Please sort vocab indexes in ascending order starting from 0"
            weights[idx, :] = get_float_encoding(token, embedding_dim, vmax)
        weights = weights.to(device=get_device())
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)
        self.vocab = vocab

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)


class IntEncoding(nn.Embedding):
    """
    A nn.Embedding inspired class to generate fixed embedding vectors that represent
    positive integers passed as tokens.
    NOTE: Tokens representing numbers need to follow notation _8_2_ to represent 80.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, vocab: Dict, *args, **kwargs
    ) -> None:
        """
        Constructor for FloatEmbedding; sets up the fixed embedding matrix.

        Args:
            num_embeddings (int): size of the dictionary of embeddings.
            embedding_dim (int): the size of each embedding vector
            vocab (Dict): the language dictionary with tokens as keys and indexes as
                values. Length needs to match num_embeddings

        Raises:
            ValueError: if num_embeddings does not match len(vocab).
            TypeError: if neither None nor a number is passed as vmax
            ValueError: if vmax is negative.
        """

        if "vmax" in kwargs.keys():
            kwargs.pop("vmax")

        super(IntEncoding, self).__init__(
            num_embeddings, embedding_dim, *args, **kwargs
        )

        if not len(vocab) == num_embeddings:
            raise ValueError(
                f"num_embeddings needs to match size of vocabulary ({num_embeddings}!={len(vocab)})"
            )

        weights = torch.zeros(num_embeddings, embedding_dim)
        for idx, (token, index) in enumerate(vocab.items()):
            assert (
                idx == index
            ), "Please sort vocab indexes in ascending order starting from 0"
            weights[idx, :] = get_int_encoding(token, embedding_dim)

        weights = weights.to(device=get_device())
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)
        self.vocab = vocab

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)
