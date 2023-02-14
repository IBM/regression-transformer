from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import transformers
from transformers import DataCollatorForPermutationLanguageModeling
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import logging

from .collator_utils import get_mask, get_permutation_order

logger = logging.get_logger(__name__)


@dataclass
class BaseCollator(DataCollatorForPermutationLanguageModeling):
    def finalize(self, batch: torch.Tensor, val: int = 0) -> torch.Tensor:
        """Sequence length has to be even for PLM collator, see:
        https://github.com/huggingface/transformers/issues/7341

        Args:
            batch (torch.Tensor): 2D Tensor (batch_size x seq_len)
            val (float): Value to fill with.

        Returns:
            torch.Tensor: 2D Tensor (batch_size x seq_len)
        """
        if batch.size(1) % 2 != 0:
            return torch.cat([batch, torch.ones(batch.size(0), 1).long() * val], axis=1)
        return batch.long()

    def attention_mask(self, batch: torch.Tensor, dropout: float = 0.0) -> torch.Tensor:
        attention_mask = (~(batch == 0)).to(float)
        return attention_mask

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch = self.finalize(self._tensorize_batch(examples))
        attention_mask = self.attention_mask(batch)

        inputs, perm_mask, target_mapping, labels = self.mask_tokens(batch)
        return {
            "input_ids": inputs,
            "perm_mask": perm_mask,
            "target_mapping": target_mapping,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def mask(
        self,
        inputs: torch.Tensor,
        masked_indices: torch.Tensor,
        labels: torch.Tensor,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        special_tokens_mask = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ],
            dtype=torch.bool,
            device=device,
        )
        masked_indices.masked_fill_(special_tokens_mask, value=0.0)

        padding_mask = labels.eq(self.tokenizer.pad_token_id)
        masked_indices.masked_fill_(padding_mask, value=0.0)

        # Mask indicating non-functional tokens, where functional tokens are [SEP], [CLS], padding, etc.
        non_func_mask = ~(padding_mask & special_tokens_mask)

        inputs[masked_indices] = self.tokenizer.mask_token_id
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        return inputs, masked_indices, non_func_mask, labels

    def find_entity_indices(
        self, inputs: torch.Tensor, entity_to_mask: Optional[int] = None
    ) -> Tuple[List, List]:
        """
        Finds the start and end indices of an entity in an input tensor

        Args:
            inputs (torch.Tensor): 2D of shape batch_size x seq_len
            entity_to_mask (int, Optional): The number of the entity to be masked.
                Defaults to None, meaning that everything after the separator token can
                be masked. If -1, then it it is randomly sampled which entity is masked.

        Returns:
            Tuple[List, List]: Two lists containing the start and end indices respectively
        """

        # Find matches for separator
        sep_matches, sep_token_pos = (inputs == self.separator_token_idx).nonzero(
            as_tuple=True
        )
        unique_sep_matches = torch.unique(sep_matches)
        assert torch.equal(
            unique_sep_matches, torch.arange(inputs.size(0))
        ), f"Found samples without separator {unique_sep_matches}"

        if entity_to_mask is None:
            # We fall back to the trivial case
            logger.debug("Masking randomly across all entities")
            # Find the separator (where textual tokens start)
            first_ent_pos = [
                int(max(sep_token_pos[torch.where(sep_matches == s)])) + 1
                for s in unique_sep_matches
            ]
            last_ent_pos = [
                int(torch.min(torch.where(s == self.tokenizer.sep_token_id)[0]) - 1)
                for s in inputs
            ]
            return first_ent_pos, last_ent_pos

        # We find the minimal number of entities contained in each sample.
        # NOTE: If any (but the last) entity itself consists of a '.'-separated sequence,
        # the masking will be incorrect. If the last entity is a '.'-separated sequence,
        # the minimum over the batch ensures that masking is correct because we just mask
        # from the k-th '.' to the end (unless all samples from the batch have a
        # '.'-separated sequence).
        # TODO: On the long run, we should set another token to separate entities
        num_entities = int((inputs == self.entity_separator_idx).sum(axis=1).min()) + 1
        if num_entities == 1 and entity_to_mask is not None:
            raise ValueError(f"Cannot mask {entity_to_mask}-th entity, only one found")

        if entity_to_mask > num_entities:
            raise ValueError(
                f"Did not find {entity_to_mask} but only {num_entities} entities."
            )

        # If -1, we fix the entity now for this batch
        if entity_to_mask == -1:
            entity_to_mask = int(torch.randint(0, num_entities, ()))

        # For each sample find the range of tokens that are considered for masking

        # Find matches for entity-separator
        entsep_matches, entsep_token_pos = (
            inputs == self.entity_separator_idx
        ).nonzero(as_tuple=True)
        unique_entsep_matches = torch.unique(entsep_matches)

        assert torch.equal(
            unique_entsep_matches, torch.arange(inputs.size(0))
        ), "Found samples with only a single entity"

        # Determine the first token that can be masked
        if entity_to_mask == 0:
            # Special case here because first entity occurs right after the separator
            # Determining the index of last separator in each sample of the batch
            first_ent_pos = [
                int(max(sep_token_pos[torch.where(sep_matches == s)])) + 1
                for s in unique_sep_matches
            ]
        else:
            # Determine index of the entity-separator (e.g., '.') that lies right
            # before the entity to be masked
            first_ent_pos = [
                int(
                    entsep_token_pos[torch.where(entsep_matches == s)][
                        entity_to_mask - 1
                    ]
                )
                + 1
                for s in unique_entsep_matches
            ]

        # Determine the last token that can be masked
        if entity_to_mask == num_entities - 1:
            # Trivial case because we can mask until the end
            last_ent_pos = [
                int(torch.min(torch.where(s == self.tokenizer.sep_token_id)[0]) - 1)
                for s in inputs
            ]
        else:
            # Determine index of the entity-separator (e.g., '.') that lies right
            # after the entity to be masked

            # fmt: off
            last_ent_pos = [
                int(entsep_token_pos[torch.where(entsep_matches == s)][entity_to_mask])
                - 1 for s in unique_entsep_matches
            ]
            # fmt: on

        return first_ent_pos, last_ent_pos


@dataclass
class SinglePropertyCollator(BaseCollator):
    """Collator class to parse samples for predicting single property."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        property_token: str,
        mask_token_order: List[int] = None,
        num_tokens_to_mask: int = -1,
        ignore_errors: bool = False,
    ):
        """
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer used for splitting.
            property_token (str): The property token of interest masked out for PP.
            num_tokens_to_mask (int, optional): How many tokens should be masked
                maximally. Defaults to -1, meaning that *all* tokens between the
                property token and the separator are masked.
            mask_token_order (List[int]): The property float/int is composed by several
                digits, e.g., 2 7 3 or 0 . 7 3 4. If only `num_tokens_to_mask` tokens
                are masked this list determines the order in which tokens are masked.
            ignore_errors (bool, optional): Whether to ignore errors during masking.
                This is useful for the self-consistency loss where it is not guaranteed
                that the model generates valid predictions. Defaults to False.

        """

        self.tokenizer = tokenizer
        if property_token not in tokenizer.vocab_list:
            raise ValueError(f"Property token {property_token} not in tokenizer vocab")

        self.property_token = property_token
        self.property_token_idx = tokenizer.vocab[property_token]

        self.num_tokens_to_mask = num_tokens_to_mask
        self.mask_token_order = mask_token_order
        self.ignore_errors = ignore_errors

        if num_tokens_to_mask > 0 and isinstance(mask_token_order, list):
            self.tokens_to_mask = torch.Tensor(
                mask_token_order[:num_tokens_to_mask]
            ).long()

        self.separator_token_idx = tokenizer.vocab[tokenizer.expression_separator]

    def mask_tokens(
        self,
        inputs: torch.Tensor,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Only need to overwrite the masking function to create batches where property
        is masked.
        The first num_tokens_to_mask tokens between property_token and separator_token
        will be masked.
        """
        logger.debug(f"Property collator masking {self.property_token}")

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for permutation language modeling. Please add a mask token if you want to use this tokenizer."
            )

        if inputs.size(1) % 2 != 0:
            raise ValueError(
                "This collator requires that sequence lengths be even to create a leakage-free perm_mask. Please see relevant comments in source code for details."
            )

        matches = []

        # Find matches for property token and separator
        for search_idx in [self.property_token_idx, self.separator_token_idx]:
            sample_matches, match_positions = (inputs == search_idx).nonzero(
                as_tuple=True
            )
            keep_idxs = list(range(inputs.size(0)))

            if list(range(inputs.size(0))) != sample_matches.tolist():
                token = self.tokenizer.convert_ids_to_tokens(search_idx)
                # Some samples might not have any property token
                if search_idx == self.property_token_idx:
                    if len(set(sample_matches.tolist())) < inputs.size(0):
                        logger.debug(f"Found sample without property token {token}")

                        match_pos = torch.zeros(inputs.size(0), device=device)
                        keep_idxs = []
                        for sample_idx in range(inputs.size(0)):
                            sample_sep_pos = match_positions[
                                sample_matches == sample_idx
                            ]
                            if len(sample_sep_pos) > 0:
                                keep_idxs.append(sample_idx)
                                match_pos[sample_idx] = torch.min(sample_sep_pos)
                    else:
                        if not self.ignore_errors:
                            raise ValueError(
                                f"Multiple occurrences of {token} not supported"
                            )
                        else:
                            s = sample_matches.tolist()
                            dups = list(set([x for x in s if s.count(x) > 1]))
                            x = "".join(
                                self.tokenizer.convert_ids_to_tokens(inputs[dups[0]])
                            )
                            logger.error(f"Multiple occurrences of property {x}")

                else:
                    # Some samples have multiple or no matches of SEPARATOR
                    match_pos = torch.zeros(inputs.size(0), device=device)
                    for sample_idx in range(inputs.size(0)):
                        # For each sample, find the first sep token that follows the property token
                        sample_sep_pos = match_positions[sample_matches == sample_idx]
                        rel_sep_pos = sample_sep_pos - matches[-1][sample_idx]
                        match_pos[sample_idx] = sample_sep_pos[
                            torch.where(rel_sep_pos > 0)[0][0]
                        ]
                    match_pos = match_pos[keep_idxs]

                match_positions = match_pos

            matches.append(match_positions)
        # Creating the mask and target_mapping tensors
        labels = inputs.clone()
        masked_indices = torch.full(labels.shape, 0, dtype=torch.bool, device=device)
        target_mapping = torch.zeros(
            (labels.size(0), labels.size(1), labels.size(1)),
            dtype=torch.float32,
            device=device,
        )
        if self.mask_token_order is None:
            # Mask *all* tokens
            start_mask = (matches[0] + 1).int()
            end_mask = matches[1].int()

            for i in range(masked_indices.shape[0]):
                masked_indices[i, start_mask[i] : end_mask[i]] = 1
                target_mapping[i] = torch.eye(labels.size(1), device=device)
        else:
            # Mask only tokens saved in tokens_to_mask
            for i in range(masked_indices.shape[0]):
                # The 1 we need as offset
                mask_tokens = matches[0][i] + 1 + self.tokens_to_mask
                # Needed since the number might have different number of decimals
                mask_tokens = mask_tokens[mask_tokens < matches[1][i]]
                masked_indices[i, mask_tokens] = 1
                target_mapping[i] = torch.eye(labels.size(1), device=device)

        inputs, masked_indices, non_func_mask, labels = self.mask(
            inputs, masked_indices, labels, device=device
        )
        perm_mask = get_permutation_order(
            labels, masked_indices, non_func_mask, device=device
        )

        return inputs, perm_mask, target_mapping, labels


@dataclass
class PropertyCollator(SinglePropertyCollator):
    """Collator class to parse samples for property prediction task.
    NOTE: This can handle multiple properties
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        property_tokens: Iterable[str],
        num_tokens_to_mask: Optional[Iterable[int]] = None,
        mask_token_order: Optional[Iterable[List[int]]] = None,
        ignore_errors: bool = False,
    ):
        """
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer used for splitting.
            property_tokens (Iterable[str]): An iterable of property tokens to be
                masked out for PP.
            num_tokens_to_mask (Iterable[int], optional): How many tokens should be masked
                maximally. Defaults to None, meaning that *all* tokens between the
                property tokens and the separator are masked.
            mask_token_order (Iterable[List[int]], optional): The property float/int is
                composed by several digits, e.g., 2 7 3 or 0 . 7 3 4. If only
                `num_tokens_to_mask` tokens are masked this list determines the order in
                which tokens are masked. If `num_tokens_to_mask` is None (or [-1]),
                this value is ignored. Defaults to None.
            ignore_errors (bool, optional): Whether to ignore errors during masking.
                This is useful for the self-consistency loss where it is not guaranteed
                that the model generates valid predictions. Defaults to False.

        """
        if num_tokens_to_mask is None:
            num_tokens_to_mask = [-1] * len(property_tokens)
        if mask_token_order is None:
            mask_token_order = [None] * len(property_tokens)

        assert len(property_tokens) == len(num_tokens_to_mask), "Lengths must match"
        assert len(property_tokens) == len(mask_token_order), "Lengths must match"

        self.tokenizer = tokenizer
        self.property_tokens = property_tokens
        self.mask_token_order = mask_token_order
        self.num_tokens_to_mask = num_tokens_to_mask

        collators = []
        for token, num_to_mask, order in zip(
            property_tokens, num_tokens_to_mask, mask_token_order
        ):
            collators.append(
                SinglePropertyCollator(
                    tokenizer=tokenizer,
                    property_token=token,
                    num_tokens_to_mask=num_to_mask,
                    mask_token_order=order,
                    ignore_errors=ignore_errors,
                )
            )
        self.collators = collators

    def mask_tokens(
        self,
        inputs: torch.Tensor,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Masking tokens, one collator at a time

        Args:
            inputs (torch.Tensor): Input tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Tensors with inputs, permutation mask, target mapping and labels
        """

        perm_mask = torch.zeros(*inputs.shape, inputs.shape[-1], device=device).bool()
        labels = torch.ones(*inputs.shape, device=device).long() * -100

        for idx, collator in enumerate(self.collators):
            inputs, _perm_mask, target_mapping, _labels = collator.mask_tokens(
                inputs, device=device
            )
            # perm_mask = torch.logical_or(perm_mask, _perm_mask).float()
            perm_mask = (perm_mask.bool() | _perm_mask.bool()).float()
            labels = torch.max(labels, _labels)
        return inputs, perm_mask, target_mapping, labels


@dataclass
class ConditionalGenerationEvaluationCollator(BaseCollator):
    """Collator class to parse samples for conditional generation task.
    NOTE: Can only handle a single property. Intended to use for evaluation to
    assess the effect of changing one property.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        property_token: str,
        conditioning_range: Iterable[Union[float, int]],
        plm_probability: float = 1 / 6,
        max_span_length: int = 5,
        entity_to_mask: Optional[int] = None,
        entity_separator_token: Optional[str] = None,
    ):
        """
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer used for splitting.
            property_token (str): The property token of interest masked out for PP.
            conditioning_range (Iterable[Union[float, int]]): An iterable of either
                floats or ints which determine the numbers used for conditioning. For a
                property between 0 and 1, this could be e.g.: [0, .3, .6, .9].
            plm_probability (float, optional): The probability for each token to be
                masked. Defaults to 1/6.
            max_span_length (int, optional): Positive integer determining the maximal
                span of consequetively masked tokens. The true span length is sampled
                uniformally from [1, max_span_length]. Defaults to 5.
            entity_to_mask (int, Optional): The entity-index on which the masking is
                done. If None, it is assumed that the text after the separator is just
                one entity.
            entity_separator_token (str, Optional): the string/token used to separate
                entities. Only necessary if `entity_to_mask` is not None.
        """

        self.tokenizer = tokenizer

        if property_token not in tokenizer.vocab_list:
            raise ValueError("Property token not in tokenizer vocab")

        self.property_token = property_token
        self.property_token_idx = tokenizer.vocab[property_token]
        self.separator_token_idx = tokenizer.vocab[tokenizer.expression_separator]

        self.plm_probability = plm_probability
        self.max_span_length = max_span_length
        self.conditioning_range = list(
            map(
                lambda x: property_token + str(x) + tokenizer.expression_separator,
                conditioning_range,
            )
        )
        self.num_primed = len(self.conditioning_range)

        self.entity_to_mask = entity_to_mask
        self.entity_separator_token = entity_separator_token
        if entity_separator_token:
            self.entity_separator_idx = self.tokenizer.convert_tokens_to_ids(
                entity_separator_token
            )

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]

        batch = self.finalize(self._tensorize_batch(examples))
        inputs, perm_mask, target_mapping, labels, true_prop = self.mask_tokens(batch)
        attention_mask = self.attention_mask(inputs)
        return {
            "input_ids": inputs,
            "perm_mask": perm_mask,
            "target_mapping": target_mapping,
            "labels": labels,
            "real_property": true_prop,
            "attention_mask": attention_mask,
        }

    def mask_tokens(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - Masks never on the property but only after the last separator
        - Creates self.num_primed instances per sample, each with a
            different primer.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for permutation language modeling. Please add a mask token if you want to use this tokenizer."
            )

        if inputs.size(1) % 2 != 0:
            raise ValueError(
                "This collator requires that sequence lengths be even to create a leakage-free perm_mask. Please see relevant comments in source code for details."
            )

        # Find matches for property token
        a, property_token_pos = (inputs == self.property_token_idx).nonzero(
            as_tuple=True
        )
        primer_start_pos = property_token_pos.tolist()
        if len(a) != inputs.size(0) or list(range(inputs.size(0))) != a.tolist():
            raise ValueError(f"Found samples with multiple or no token counts {inputs}")

        # Find matches for separator token
        b, separator_token_pos = (inputs == self.separator_token_idx).nonzero(
            as_tuple=True
        )

        # Find the next separator after the property token
        primer_end_pos = []
        for i, p in enumerate(primer_start_pos):
            cans = separator_token_pos[b == i]
            primer_end_pos.append(int(cans[torch.min(torch.where(cans - p > 0)[0])]))

        # Creating the mask and target_mapping tensors
        labels = inputs.clone()

        # Determine which token (positions) are considered for masking
        first_ent_pos, last_ent_pos = self.find_entity_indices(
            inputs, self.entity_to_mask
        )

        # Start masking from the beginning of the SMILES/SELFIES
        masked_indices, target_mapping = get_mask(
            labels,
            max_span_length=self.max_span_length,
            plm_probability=self.plm_probability,
            mask_start_idxs=first_ent_pos,
            mask_end_idxs=last_ent_pos,
        )

        inputs, masked_indices, non_func_mask, labels = self.mask(
            inputs, masked_indices, labels
        )

        perm_mask = get_permutation_order(labels, masked_indices, non_func_mask)
        primed_inputs, real_property = self.prime_inputs(
            inputs, primer_start_pos, primer_end_pos
        )

        # Repeat all remaining output variables for conditioning
        perm_mask = perm_mask.repeat_interleave(self.num_primed, dim=0)
        target_mapping = target_mapping.repeat_interleave(self.num_primed, dim=0)
        labels = labels.repeat_interleave(self.num_primed, dim=0)
        return primed_inputs.long(), perm_mask, target_mapping, labels, real_property

    def prime_inputs(
        self, inputs: torch.Tensor, property_token_pos: List, separator_token_pos: List
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replicates all sample of a batch k times with k different primers for the
        property specified in constructor and the k different primers specified in
        conditioning_range.
        NOTE: Can not handle multiple properties at the moment.

        Args:
            inputs (torch.Tensor): Input tensor with masked tokens
            property_token_pos (List): List of length inputs.size(0) denoting the
                index for the property_token in each sample.
            separator_token_pos (List): List of length inputs.size(0) denoting the
                index for the separator_token in each sample.dataclass

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                primed_inputs (torch.Tensor): of shape inputs.size(0) * self.num_primed
                real_property (torch.Tensor): of shape input.size(0) holding the true
                    property value for each sample.
        """
        # Duplicate inputs with different primers
        primed_inputs = torch.zeros(self.num_primed * inputs.shape[0], inputs.shape[1])
        real_property = torch.zeros(inputs.shape[0])
        for i in range(inputs.shape[0]):
            # Maintain and return true property
            real_property[i] = self.tokenizer.floating_tokens_to_float(
                self.tokenizer.decode(
                    inputs[i, property_token_pos[i] + 1 : separator_token_pos[i]]
                ).split(" ")
            )

            for pidx, primer in enumerate(self.conditioning_range):
                primed_inputs[(i * self.num_primed) + pidx, :] = inputs[i]
                keep = self.tokenizer(primer)["input_ids"][2:-2]
                primed_inputs[
                    (i * self.num_primed) + pidx,
                    property_token_pos[i] + 1 : property_token_pos[i] + 1 + len(keep),
                ] = torch.Tensor(keep)

        return primed_inputs, real_property


@dataclass
class ConditionalGenerationTrainCollator(BaseCollator):
    """Collator class that does not mask the properties but anything else as a
    regular DataCollatorForPermutationLanguageModeling. Can optionally replace
    the properties with sampled values.
    NOTE: This collator can deal with multiple properties.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        property_tokens: Iterable[str],
        plm_probability: float = 1 / 6,
        max_span_length: int = 5,
        do_sample: bool = False,
        property_value_ranges: Optional[Iterable[Iterable[float]]] = None,
        **kwargs,
    ):
        """
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer used for splitting.
            property_tokens (Iterable[str]): An iterable of property tokens which will
                be altered (if do_sample=True) or kept (if do_sample=False).
             NOTE: All samples should have same amount of properties, in same order.
             NOTE: Order of property_tokens has to match order of samples in dataset.
            plm_probability (float, optional): The probability for each token to be
                masked. Defaults to 1/6.
            max_span_length (int, optional): Positive integer determining the maximal
                span of consequetively masked tokens. The true span length is sampled
                uniformally from [1, max_span_length]. Defaults to 5.
            do_sample (bool, optional): Whether or not the properties should be
                left untouched (default) or imputed for conditional generation.
            property_value_ranges (Iterable, optional): Only needed if do_sample
             is True. In this case this should be an array of length equal to number
             of properties in dataset (which should be constant for all samples!). The
             two values in each item of the list are interpreted as min and max for
             the uniform property sampling.
            **kwargs: Ignored keyword arguments for child class compatibility.
        """

        self.tokenizer = tokenizer
        self.separator_token_idx = tokenizer.vocab[tokenizer.expression_separator]
        self.plm_probability = plm_probability
        self.max_span_length = max_span_length
        self.do_sample = do_sample

        self.property_tokens = property_tokens
        self.num_props = len(property_tokens)
        self.property_token_idxs = [tokenizer.vocab[p] for p in property_tokens]

        if self.do_sample:
            if not property_value_ranges:
                raise TypeError("property_value_ranges not provided for sampling.")

            self.property_value_ranges = property_value_ranges
            assert self.num_props == len(property_value_ranges), "Ranges missing"
            for pvr in property_value_ranges:
                if not (pvr[1] > pvr[0]):
                    raise ValueError(f"Property range should be sorted ({pvr})")

            # Create sampling functions (the x is a fixed parameter for each function)
            # Functions still need an argument (_), but it is ignored here. It is
            # needed to avoid the need of overwriting `sample_property` in child classes
            self.sampling_functions = [
                lambda _, x=pr: float((x[0] - x[1]) * torch.rand(1) + x[1])
                for pr in property_value_ranges
            ]

            # For compatibility with related classes
            self.num_primed = 1

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch = self.finalize(self._tensorize_batch(examples))
        attention_mask = self.attention_mask(batch)
        inputs, perm_mask, target_mapping, labels = self.mask_tokens(batch)
        inputs, true_prop, sample_weights = self.sample_property(inputs)

        return {
            "input_ids": inputs,
            "perm_mask": perm_mask,
            "target_mapping": target_mapping,
            "labels": labels,
            "real_property": true_prop,
            "sample_weights": sample_weights,
            "attention_mask": attention_mask,
        }

    def mask_tokens(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Token masking function. Masks never on the property but only after the
        last separator, i.e., on the molecule.

        Args:
            inputs (torch.Tensor): Input tensor of shape batch_size x seq_len

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                inputs: Long tensor of shape batch_size x seq_len with some tokens
                    being masked.
                perm_mask: Tensor of shape batch
                target_mapping: Tensor of shape batch_size x seq_len x seq_len that
                    indicates for each position which is its target. Always a diagonal
                    matrix.
                labels: Tensor of shape batch_size x seq_len with true tokens at masked
                    positions and -100 everywhere else.
        """
        logger.debug("Using CG Collator")

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for permutation language modeling. Please add a mask token if you want to use this tokenizer."
            )

        if inputs.size(1) % 2 != 0:
            raise ValueError(
                "This collator requires that sequence lengths be even to create a leakage-free perm_mask. Please see relevant comments in source code for details."
            )

        # Find matches for separator
        separator_matches, separator_token_pos = (
            inputs == self.separator_token_idx
        ).nonzero(as_tuple=True)
        unique_separator_matches = torch.unique(separator_matches)

        assert torch.equal(
            unique_separator_matches, torch.arange(inputs.size(0))
        ), f"Found samples without separator {unique_separator_matches}"

        # Retrieve index of last occurrence of separator and add one to get
        # first index which could be masked
        last_sep_pos = [
            int(max(separator_token_pos[torch.where(separator_matches == s)])) + 1
            for s in unique_separator_matches
        ]

        # Creating the mask and target_mapping tensors
        labels = inputs.clone()

        # Start masking from the beginning of the SMILES/SELFIES
        masked_indices, target_mapping = get_mask(
            labels,
            max_span_length=self.max_span_length,
            plm_probability=self.plm_probability,
            mask_start_idxs=last_sep_pos,
        )

        inputs, masked_indices, non_func_mask, labels = self.mask(
            inputs, masked_indices, labels
        )

        perm_mask = get_permutation_order(labels, masked_indices, non_func_mask)

        return inputs, perm_mask, target_mapping, labels

    def sample_property(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a primer for each property, fills the input with the new primer and
        saves the real property values.

        NOTE: If self.do_sample is False, no sampling occurrs, but only the real
            property values are extracted.

        Args:
            inputs (torch.Tensor): Long tensor of shape batch_size x seq_len

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                inputs: Long tensor of shape batch_size x seq_len, optionally with
                    property values replaced by primers.
                real_property: Float tensor of shape batch_size x num_props holding
                    the true property values.
                sample_weights: Float tensor of shape batch_size holding the weights
                    for each sample. Scaled between -1 and 1.
        """

        # Find matches for separator
        separator_matches, separator_token_pos = (
            inputs == self.separator_token_idx
        ).nonzero(as_tuple=True)

        # Replace the true properties if applicable
        real_property = torch.zeros(inputs.shape[0], self.num_props)
        sample_weights = torch.ones(inputs.shape[0], self.num_props)
        # For each property
        for pidx, (prop_token, prop_token_idx) in enumerate(
            zip(self.property_tokens, self.property_token_idxs)
        ):
            self.pidx = pidx

            # Set property values/ranges
            if self.do_sample:
                sample = self.sampling_functions[pidx]
                prop_values = self.property_value_ranges[pidx]
                property_range = abs(prop_values[1] - prop_values[0])

            # Find matches for property tokens
            property_matches, property_token_pos = (inputs == prop_token_idx).nonzero(
                as_tuple=True
            )
            if len(set(property_matches.tolist())) < inputs.size(0):
                logger.debug(f"Found samples without property token {prop_token}")
            elif len(property_matches.tolist()) > len(set(property_matches.tolist())):
                raise ValueError(f"Multiple occurrences of {prop_token} not supported.")

            # Mask from property til next separator (samples w/o property are not masked)
            for i, (prow, pcol) in enumerate(zip(property_matches, property_token_pos)):
                # First, extract true property

                # Find all separators in current sample, then find the next one
                separator_idxs = separator_token_pos[
                    torch.where(separator_matches == prow)
                ]
                sep_idx = min([x for x in separator_idxs if x > pcol])

                realprop = self.tokenizer.floating_tokens_to_float(
                    self.tokenizer.decode(inputs[prow, pcol + 1 : sep_idx]).split(" ")
                )
                real_property[prow, pidx] = realprop
                if self.do_sample:
                    # Sample property and fill it in
                    num_digits = sep_idx - (pcol + 1)
                    sampled_prop = sample(realprop)  # ignored in the base class
                    sampled_prop_str = (
                        prop_token
                        + str(sampled_prop)[:num_digits]
                        + self.tokenizer.expression_separator
                    )
                    new_tokens = torch.Tensor(
                        self.tokenizer(sampled_prop_str)["input_ids"]
                    )
                    new_tokens = new_tokens[
                        2 : torch.where(new_tokens == self.separator_token_idx)[0]
                    ]
                    inputs[prow, pcol + 1 : sep_idx] = new_tokens
                    # Compute a sample weight between -1 and 1. 1 means sampled was very
                    # similar to real property, -1 means it was very distinct. Can
                    # be used to scale the loss of each sample.
                    sample_weights[prow, pidx] = self.get_sample_weight(
                        realprop, sampled_prop, property_range
                    )

        return inputs, real_property, sample_weights.mean(axis=-1)

    def get_sample_weight(
        self, real_prop: float, sampled_prop: float, prop_range: float
    ) -> float:
        """Function to generate the sample weight for the loss function. Based on the
        assumption that if a primer close to the real property was sampled, sample weight
        is high and if a very distant primer was sampled, the sample weight is negative,
        s.t., the model will *maximize* the BCE loss.
        NOTE: Generally, this will result in most weights being positive.

        Args:
            real_prop (float): Property score of sample
            sampled_prop (float): Primer (samplzed property) of sample
            prop_range (float): Range of possible properties (max-min).

        Returns:
            float: Sample weight.
        """
        return 2 * ((-1 * (abs(real_prop - sampled_prop) / prop_range)) + 1) - 1


class MultiEntityCGTrainCollator(ConditionalGenerationTrainCollator):
    """
    A training collator the conditional-generation task that can handle multiple entities
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        property_tokens: Iterable[str],
        plm_probability: float = 1 / 6,
        max_span_length: int = 5,
        do_sample: bool = False,
        property_value_ranges: Optional[Iterable[Iterable[float]]] = None,
        entity_separator_token: str = ".",
        mask_entity_separator: bool = False,
        entity_to_mask: int = -1,
    ):
        """
        For arguments before `entity_separator_token`, check out the parent class.

        Args:
            entity_separator_token (str, optional): The token that is used to separate
                entities in the input. Defaults to '.' (applicable to SMILES & SELFIES)
            mask_entity_separator (bool, optional): Whether or not the entity separator
                token can be masked. If True, *all** textual tokens can be masked and we
                just default to the parent class behavior. If False, the exact behavior
                depends on the entity_to_mask argument. Defaults to False.
            entity_to_mask (int): The entity that is being masked during
                training. 0 corresponds to first entity and so on. -1 corresponds to
                a random sampling scheme where the entity-to-be-masked is determined
                at runtime in the collator. NOTE: If `mask_entity_separator` is true,
                this argument will not have any effect. Defaults to -1.
        """
        # Set up base collator
        super().__init__(
            tokenizer=tokenizer,
            property_tokens=property_tokens,
            plm_probability=plm_probability,
            max_span_length=max_span_length,
            do_sample=do_sample,
            property_value_ranges=property_value_ranges,
        )

        self.entity_separator_token = entity_separator_token
        self.entity_separator_idx = self.tokenizer.convert_tokens_to_ids(
            entity_separator_token
        )
        self.mask_entity_separator = mask_entity_separator
        self.entity_to_mask = entity_to_mask

        if mask_entity_separator:
            # Collapse to parent class behavior
            self.mask_tokens = super().mask_tokens
            return

        self.mask_tokens = self.set_mask_tokens_fn()

    def set_mask_tokens_fn(self):
        """
        Sets the function used to mask the tokens
        """

        def mask_tokens(
            inputs: torch.Tensor,
        ) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            """
            Token masking function. Masks never on the property but only after the
            last separator, i.e., on the molecule. Will mask only one on entity per
            batch, dependent on the `entity_to_mask` attribute.

            NOTE: For remaining docstring see `mask_tokens` in parent class.
            """

            if self.tokenizer.mask_token is None:
                raise ValueError(
                    "This tokenizer does not have a mask token which is necessary for permutation language modeling. Please add a mask token if you want to use this tokenizer."
                )

            if inputs.size(1) % 2 != 0:
                raise ValueError(
                    "This collator requires that sequence lengths be even to create a leakage-free perm_mask. Please see relevant comments in source code for details."
                )

            # Creating the mask and target_mapping tensors
            labels = inputs.clone()

            first_ent_pos, last_ent_pos = self.find_entity_indices(
                inputs, self.entity_to_mask
            )

            # Start masking from the beginning of the SMILES/SELFIES
            masked_indices, target_mapping = get_mask(
                labels,
                max_span_length=self.max_span_length,
                plm_probability=self.plm_probability,
                mask_start_idxs=first_ent_pos,
                mask_end_idxs=last_ent_pos,
            )

            inputs, masked_indices, non_func_mask, labels = self.mask(
                inputs, masked_indices, labels
            )
            logger.debug(f"Multientity CG inputs {inputs[0,:]}")

            perm_mask = get_permutation_order(labels, masked_indices, non_func_mask)

            return inputs, perm_mask, target_mapping, labels

        return mask_tokens


TRAIN_COLLATORS = {
    "property": PropertyCollator,
    "vanilla_cg": ConditionalGenerationTrainCollator,
    "multientity_cg": MultiEntityCGTrainCollator,
}
EVAL_COLLATORS = {
    "property": PropertyCollator,
    "conditional_generation": ConditionalGenerationEvaluationCollator,
}
