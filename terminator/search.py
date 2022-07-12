"""Decoding utilities."""
from math import log
from sys import float_info

import numpy as np
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
    """ "Greedy search."""

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
    """ "Sampling search."""

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


class BeamSearch(Search):
    """Beam search."""

    def __init__(
        self, beam_width: int = 3, temperature: float = 1.0, top_tokens: int = 5
    ):
        """
        Initialize the beam search.
        Args:
            beam_width (int, optional): top sequences returned. Defaults to 3.
            temperature (float, optional): temperature parameter. Defaults to
                1.0, a.k.a., no temperature. Temperature < 1 results in a more
                descriminative softmax, > 1 in a flatter distribution.
            top_tokens (int, optional): number of top dictionary tokens kept
                for the search, defaults to 5.
        """
        super().__init__()
        self.beam_width = beam_width
        self.temperature = temperature
        self.top_tokens = top_tokens

    def _beam_step_per_sequence(self, probabilities: torch.Tensor, beams: list) -> list:
        """
        Perform a beam search step.
        Args:
            probabilities (torch.Tensor): probabilities for the current step.
                (beam_width, vocabulary_size).
            beams (list): beams containg sequence and score. Length is equal
                to beam_width.
        Returns:
            list: updated beams.
        """
        all_candidates = list()
        # expand each current candidate
        for probability, beam in zip(probabilities, beams):
            a_sequence, score = beam
            # Sort the probabilities over dict and select indices of top n
            top_token_indexes = np.argsort(-probability)[: self.top_tokens]
            for top_token in top_token_indexes:
                candidate = [
                    a_sequence + [top_token],
                    score + log(probability[top_token] + float_info.epsilon),
                ]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda pair: pair[1], reverse=True)
        # select best
        return ordered[: self.beam_width]

    def _beam_per_sequence(self, logits: torch.Tensor) -> tuple:
        """
        Beam per sequence in the batch.
        Args:
            logits (torch.Tensor): logits.
                (length, vocabulary_size)
        Returns:
            tuple: a tuple containing:
            - a tensor with tokens. (length, beam_width)
            - score. (beam_width)
        """
        beams = [[list(), 0.0]]
        probabilities = torch.softmax(logits.div(self.temperature), 1)
        # walk over each step in sequence
        for probability in probabilities:
            probability_beams = torch.stack(
                [probability] + [probability.clone() for _ in range(self.beam_width)]
            )
            beams = self._beam_step_per_sequence(probability_beams, beams)
        sequences, scores = zip(*beams)
        return (torch.tensor(list(sequences)).T, torch.tensor(list(scores)))

    def forward(self, logits: torch.Tensor) -> tuple:
        """
        Perform the beam search for a non-autoregressive generator.
        Args:
            logits (torch.Tensor): the model's
                logits. (batch_size, length, vocabulary_size)
        Returns:
            tuple: a tuple containing:
            - the token indexes for each top sequence.
                (batch_size, length, beam_width)
            - scores. (batch_size, beam_width)
        """
        super().forward(logits)
        tokens, scores = zip(
            *[self._beam_per_sequence(sequence) for sequence in logits]
        )
        return (torch.stack(tokens), torch.stack(scores))

    def step(self, logits: torch.Tensor, beams: list) -> tuple:
        """
        Perform a single beam search step for an autoregressive model.
        Args:
            logits (torch.Tensor): the model's
                logits. (beam_width, batch_size, vocabulary_size)
            beams (list): beams for all the batch.
        Returns:
            tuple: a tuple containing:
            - the token indexes for all the batch.
                (beam_width, batch_size)
            - updated beams for all the batch.
        """
        super().step(logits)
        probabilities = torch.softmax(logits.div(self.temperature), 2)
        updated_beams = [
            self._beam_step_per_sequence(sample_probability, sample_beams)
            for sample_probability, sample_beams in zip(
                probabilities.permute(1, 0, 2), beams
            )
        ]
        token_beams = (
            torch.stack(
                [
                    # get last token for each beam
                    torch.tensor([beam[0][-1] for beam in sample_beams])
                    for sample_beams in updated_beams
                ]
            )
            .permute(1, 0)
            .to(self.device)
        )
        return (token_beams, updated_beams)


SEARCH_FACTORY = {"greedy": GreedySearch, "beam": BeamSearch, "sample": SamplingSearch}
