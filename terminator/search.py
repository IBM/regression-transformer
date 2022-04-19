"""Decoding utilities."""
import torch
import transformers
from torch import nn

from .utils import get_device


class Search(nn.Module):
    """Base search class."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = get_device()

    def forward(self, logits: torch.Tensor) -> object:
        """
        Error handling.

        Args:
            logits: torch.Tensor (Tensor): the model's
                logits. (batch_size, length, vocabulary_size)
        Returns:
            object: the search output.
        """
        if not len(logits.shape) == 3:
            raise ValueError(f"Logits need to be 3D Tensor, was: {logits.shape}")
        if not type(logits) == torch.Tensor:
            raise TypeError(f"Logits need to be torch.Tensor, was: {type(logits)}")

    def step(self, logits: torch.Tensor) -> object:
        """
        Error handling.

        Args:
            logits: torch.Tensor (Tensor): the model's
                logits. (batch_size, vocabulary_size)
        Returns:
            object: the search output.
        """
        if len(logits.shape) > 3:
            raise ValueError(f"Logits need to be 2D or 3D Tensor, was: {logits.shape}")
        if not type(logits) == torch.Tensor:
            raise TypeError(f"Logits need to be torch.Tensor, was: {type(logits)}")


class GreedySearch(Search):
    """"Greedy search."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform the greedy search.

        Args:
            logits: torch.Tensor (Tensor): the model's
                logits. (batch_size, length, vocabulary_size)
        Returns:
            torch.Tensor: the token indexes selected. (batch_size, length)
        """
        super().forward(logits)

        return torch.argmax(logits, 2)

    def step(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform a greedy search step.

        Args:
            logits (torch.Tensor): the model's
                logits. (batch_size, vocabulary_size)
        Returns:
            torch.Tensor: the token indexes for all the batch. (batch_size, 1).
        """
        super().step(logits)
        return torch.argmax(logits, 1, keepdim=True)


class SamplingSearch(Search):
    """"Sampling search."""

    def __init__(self, temperature: float = 1.0, *args, **kwargs):
        """
        Initialize the sampling search.

        Args:
            temperature (float, optional): temperature parameter. Defaults to
                1.0, a.k.a., no temperature. Temperature < 1 results in a more
                descriminative softmax, > 1 in a flatter distribution.
        """
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform the sampling search.

        Args:
            logits: torch.Tensor (Tensor): the model's
                logits. (batch_size, length, vocabulary_size)
        Returns:
            torch.Tensor: the token indexes selected. (batch_size, length)
        """
        super().forward(logits)
        probabilities = torch.softmax(logits.div(self.temperature), 2)
        return torch.stack(
            [torch.multinomial(probability, 1) for probability in probabilities]
        ).squeeze(dim=-1)

    def step(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform a sampling search step.

        Args:
            logits (torch.Tensor): the model's
                logits. (batch_size, vocabulary_size)
        Returns:
            torch.Tensor: the token indexes for all the batch. (batch_size, 1).
        """
        super().step(logits)
        probabilities = torch.softmax(logits.div(self.temperature), 1)
        return torch.stack(
            [torch.multinomial(probability, 1) for probability in probabilities]
        )


SEARCH_FACTORY = {"greedy": GreedySearch, "sample": SamplingSearch}
