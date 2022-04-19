from typing import Optional

import torch
import transformers


def get_mask(
    labels: torch.Tensor,
    max_span_length: int,
    plm_probability: float,
    mask_start_idxs: Optional[torch.Tensor] = None,
    mask_end_idxs: Optional[torch.Tensor] = None,
) -> (torch.Tensor, torch.Tensor):
    """Receives a tensor of labels and computes the masked_indices and the target
        mapping.

    Args:
        labels (torch.Tensor): Input tensor (2D)
        max_span_length (int): Maximal length for the span of masked tokens
        plm_probability (float): Probability for each token to be masked.
        mask_start_idxs (torch.Tensor, Optional): Tensor of length labels with indices
            for first possible token to mask.
        mask_end_idxs (torch.Tensor, Optional): Tensor of length labels with indices
            for last possible token to mask.

    Returns:
        masked_indices: 2D Tensor of masked indices.
        target_mapping: 3D tensor of diagonal matrices for each sample.
    """

    # Creating the mask and target_mapping tensors
    masked_indices = torch.full(labels.shape, 0, dtype=torch.bool)
    target_mapping = torch.zeros(
        (labels.size(0), labels.size(1), labels.size(1)), dtype=torch.float32
    )
    # If on-/offset for masking are not provided we can mask from start to end
    if mask_start_idxs is None:
        mask_start_idxs = [0] * labels.size(0)
    if mask_end_idxs is None:
        mask_end_idxs = [1 * labels.size(1)] * labels.size(0)

    for i in range(labels.size(0)):
        # Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
        cur_len = mask_start_idxs[i]
        max_len = mask_end_idxs[i]

        # If the masking range is just a single token, we always mask it
        if cur_len == max_len:
            masked_indices[i, cur_len] = 1

        while cur_len < max_len:
            # Sample (length of span of tokens to be masked), take the minimum to avoid
            # that the span length is longer than the molecule length
            span_length = min(
                torch.randint(1, max_span_length + 1, (1,)).item(), max_len - cur_len
            )
            # Reserve a context of length `context_length = span_length / plm_probability` to surround the span to be masked
            context_length = int(span_length / plm_probability)
            # Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length - span_length]` and mask tokens `start_index:start_index + span_length`
            # the min is needed to avoid that the span extends over max_len
            # the max is needed to avoid that the span starts before cur_len
            start_index = max(
                min(
                    cur_len
                    + torch.randint(context_length - span_length + 1, (1,)).item(),
                    max_len - span_length,
                ),
                cur_len,
            )
            masked_indices[i, start_index : start_index + span_length] = 1
            # Set `cur_len = cur_len + context_length`
            cur_len += context_length

        # Since we're replacing non-masked tokens with -100 in the labels tensor instead of skipping them altogether,
        # the i-th predict corresponds to the i-th token.
        target_mapping[i] = torch.eye(labels.size(1))

    return masked_indices, target_mapping


def get_permutation_order(
    labels: torch.Tensor,
    masked_indices: torch.Tensor,
    non_func_mask: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:

    perm_mask = torch.zeros(
        (labels.size(0), labels.size(1), labels.size(1)),
        dtype=torch.float32,
        device=device,
    )

    for i in range(labels.size(0)):
        # Generate permutation indices i.e. sample a random factorisation order for the sequence. This will
        # determine which tokens a given token can attend to (encoded in `perm_mask`).
        # Note: Length of token sequence being permuted has to be less than or equal to reused sequence length
        # (see documentation for `mems`), otherwise information may leak through due to reuse. In this implementation,
        # we assume that reused length is half of sequence length and permutation length is equal to reused length.
        # This requires that the sequence length be even.

        # Create a linear factorisation order
        perm_index = torch.arange(labels.size(1), device=device)
        # Split this into two halves, assuming that half the sequence is reused each time
        perm_index = perm_index.reshape((-1, labels.size(1) // 2)).transpose(0, 1)
        # Permute the two halves such that they do not cross over
        perm_index = perm_index[torch.randperm(labels.size(1) // 2)]
        # Flatten this out into the desired permuted factorisation order
        perm_index = torch.flatten(perm_index.transpose(0, 1))
        # Set the permutation indices of non-masked (non-functional) tokens to the
        # smallest index (-1) so that:
        # (1) They can be seen by all other positions
        # (2) They cannot see masked positions, so there won't be information leak
        perm_index.masked_fill_(~masked_indices[i] & non_func_mask[i], -1)
        # The logic for whether the i-th token can attend on the j-th token based on the factorisation order:
        # 0 (can attend): If perm_index[i] > perm_index[j] or j is neither masked nor a functional token
        # 1 (cannot attend): If perm_index[i] <= perm_index[j] and j is either masked or a functional token
        perm_mask[i] = (
            perm_index.reshape((labels.size(1), 1))
            <= perm_index.reshape((1, labels.size(1)))
        ) & masked_indices[i]

    return perm_mask
