"""Tokenization utilties for exrepssions."""

import logging
import re
import sys
from typing import Dict, List, Set, Tuple

import torch
import transformers
from transformers import BertTokenizer

from .selfies import decoder, split_selfies

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
POLYMER_GRAPH_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|A|B|C|D|E|R|Q|Z|;|<|>|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"


class RegexTokenizer:
    """Run regex tokenization"""

    def __init__(self, regex_pattern: str) -> None:
        """Constructs a RegexTokenizer.

        Args:
            regex_pattern: regex pattern used for tokenization
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text: str) -> List[str]:
        """Regex tokenization.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        tokens = [token for token in self.regex.findall(text)]
        return tokens


class PropertyTokenizer:
    """Run a property tokenization."""

    def __init__(self) -> None:
        """Constructs a PropertyTokenizer."""
        self.regex = re.compile(r"\s*(<\w+>)\s*?(\+|-)?(\d+)(\.)?(\d+)?\s*")

    def tokenize(self, text: str) -> List[str]:
        """Tokenization of a property.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        tokens = []
        matched = self.regex.match(text)
        if matched:
            property_name, sign, units, dot, decimals = matched.groups()
            tokens = [property_name]
            if sign:
                tokens += [f"_{sign}_"]
            tokens += [
                f"_{number}_{position}_" for position, number in enumerate(units[::-1])
            ][::-1]
            if dot:
                tokens += [f"_{dot}_"]
            if decimals:
                tokens += [
                    f"_{number}_-{position}_"
                    for position, number in enumerate(decimals, 1)
                ]
        return tokens


class CharacterTokenizer:
    def __init__(self) -> None:
        """Constructs a tokenizer that simply splits each character"""
        self.tokenizer = lambda x: list(x)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize an expression.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        return self.tokenizer(text)


class SelfiesTokenizer(CharacterTokenizer):
    def __init__(self) -> None:
        """Constructs an expression tokenizer for SELFIES

        Args:
            expression_tokenizer: Separator token for properties and molecule.
                Defaults to '|'.
        """
        self.tokenizer = self.tokenize_selfies

    def tokenize_selfies(self, selfies: str) -> List[str]:
        """Tokenize SELFIES.

        NOTE: Code adapted from selfies package (`def selfies_to_hot`):
            https://github.com/aspuru-guzik-group/selfies

        Args:
            selfies (str): a SELFIES representation (character-level).

        Returns:
            Tokens: the tokenized SELFIES.
        """
        logger.debug(
            "tokenize_selfies might differ from selfies new internal `split_selfies` method"
        )
        try:
            if "." not in selfies:
                # Canonical case (only single entity)
                selfies_char_list_pre = selfies[1:-1].split("][")
                return [
                    "[" + selfies_element + "]"
                    for selfies_element in selfies_char_list_pre
                ]

            # Multiple entities present
            splitted = []
            for selfie in selfies.split("."):
                selfies_char_list_pre = selfie[1:-1].split("][")
                split_selfie = ["[" + se + "]" for se in selfies_char_list_pre]
                splitted.extend(split_selfie)
                splitted.append(".")

            return splitted[:-1]
        except Exception:
            logger.warning(f"Error in tokenizing {selfies}. Returning empty list.")
            return [""]


class ReactionSmilesTokenizer(CharacterTokenizer):
    def __init__(self, precursor_separator: str = "<energy>") -> None:
        """
        Constructs an expression tokenizer for reaction SMILES.

        Args:
            precursor_separator: a token that separates different precursors from
                another. Defaults to '<energy>'.

        """
        self.precursor_separator = precursor_separator
        self.tokenizer = RegexTokenizer(regex_pattern=SMILES_TOKENIZER_PATTERN)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize an expression.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        return self.tokenizer.tokenize(text)


class ReactionSelfiesTokenizer(SelfiesTokenizer):
    def __init__(self, precursor_separator: str = ".") -> None:
        """Constructs an expression tokenizer for SELFIES
        Args:
            expression_tokenizer: Separator token for properties and molecule.
                Defaults to '|'.
            precursor_separator: a token that separates different precursors from
                another. Defaults to '.'.
        """
        self.tokenizer = self.tokenize_reaction
        self.precursor_separator = precursor_separator

    def tokenize_reaction(self, x: str) -> List[str]:
        """Tokenize a SELFIES reaction.
        Args:
            text: text to tokenize.
        Returns:
            extracted tokens.
        """
        if self.precursor_separator in x:
            if ">>" not in x:
                logger.error(f"Pass sequence with reaction arrow: {x}")
                return []
            prec, post = x.split(">>")
            pre_tokens = []
            for pre in prec.split(self.precursor_separator):
                if pre_tokens != []:
                    pre_tokens.append(self.precursor_separator)
                # Otherwise empty molecules ('.') are simply skipped.
                if pre == ".":
                    pre_tokens.append(".")
                else:
                    pre_tokens.extend(list(split_selfies(pre)))

            post_tokens = []
            for pos in post.split(self.precursor_separator):
                if post_tokens != []:
                    post_tokens.append(self.precursor_separator)
                if pos == ".":
                    post_tokens.append(".")
                else:
                    post_tokens.extend(list(split_selfies(pos)))

        else:
            if ">>" not in x:
                return list(split_selfies(x))
            pre, post = x.split(">>")
            pre_tokens = list(split_selfies(pre))
            post_tokens = list(split_selfies(post))
        return pre_tokens + [">>"] + post_tokens


class PolymerGraphTokenizer:
    def __init__(self) -> None:
        """Constructs a tokenizer for processing string representations of Polymers"""

        # split into units of <...> but skip over ->
        self.tokenizer = RegexTokenizer(regex_pattern="<.*?[^-]>")
        self.node_tokenizer = RegexTokenizer(
            regex_pattern=POLYMER_GRAPH_TOKENIZER_PATTERN
        )

    def tokenize(self, text: str) -> List[str]:
        """Tokenize an expression.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        tokens = []
        for node in self.tokenizer.tokenize(text):
            for block in node.split("->"):
                tokens.extend(self.node_tokenizer.tokenize(block))
                tokens.append("->")
            del tokens[-1]
        return tokens


class ExpressionTokenizer:
    def __init__(
        self, expression_tokenizer: str = "|", language: str = "SMILES"
    ) -> None:
        """Constructs an expression tokenizer.

        Args:
            expression_tokenizer (str): Token separating the property. Defaults to '|'.
                Must not occur in the language itself.
            language (str): Identifier for the (chemical) language. Should be either
                'SMILES', 'SELFIES' or 'AAS'.
        """
        self.language = language
        if language == "SMILES":
            self.text_tokenizer = RegexTokenizer(regex_pattern=SMILES_TOKENIZER_PATTERN)
        elif language == "SELFIES":
            self.text_tokenizer = SelfiesTokenizer()
        elif language == "AAS":
            self.text_tokenizer = CharacterTokenizer()
        elif language == "REACTION_SMILES":
            self.text_tokenizer = PolymerGraphTokenizer()
        elif language == "REACTION_SELFIES":
            self.text_tokenizer = ReactionSelfiesTokenizer()
        elif language == "Polymer":
            self.text_tokenizer = CharacterTokenizer()
        else:
            raise ValueError(
                f"Unsupported language {language}, choose 'SMILES', 'SELFIES', 'AAS', 'REACTION_SMILES' or 'Polymer'"
            )
        self.property_tokenizer = PropertyTokenizer()
        self.expression_separator = expression_tokenizer

    def tokenize(self, text: str) -> List[str]:
        """Tokenize an expression.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        splitted_expression = text.split(self.expression_separator)
        tokens = []
        for property_expression in splitted_expression[:-1]:
            tokens.extend(self.property_tokenizer.tokenize(property_expression))
            tokens.append(self.expression_separator)
        tokens.extend(self.text_tokenizer.tokenize(splitted_expression[-1]))
        return tokens


class ExpressionBertTokenizer(BertTokenizer):
    """
    Constructs a ExpressionBertTokenizer.
    Adapted from https://github.com/huggingface/transformers

    Args:
        vocab_file: path to a token per line vocabulary file.
    """

    def __init__(
        self,
        vocab_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        pad_even: bool = True,
        language: str = "SMILES",
        precursor_separator: str = "<energy>",
        **kwargs,
    ) -> None:
        """Constructs an ExpressionTokenizer.

        Args:
            vocab_file: vocabulary file containing tokens.
            unk_token: unknown token. Defaults to "[UNK]".
            sep_token: separator token. Defaults to "[SEP]".
            pad_token: pad token. Defaults to "[PAD]".
            cls_token: cls token. Defaults to "[CLS]".
            mask_token: mask token. Defaults to "[MASK]".
            pad_even (bool): Boolean indicating whether sequences of odd length should
                be padded to have an even length. Neede for PLM in XLNet. Defaults to
                True.
            language (str): Identifier for the (chemical) language. Should be either
                'SMILES', 'SELFIES' or 'AAS', 'REACTION_SMILES' or 'Polymer'.
            precursor_separator: a token that separates different precursors from
                another. Used only for REACTION sequences. Defaults to '<energy>'.
        """
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=False,
            do_basic_tokenize=True,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        # define tokenization utilities
        self.language = language
        if language == "SMILES":
            self.text_tokenizer = RegexTokenizer(regex_pattern=SMILES_TOKENIZER_PATTERN)
        elif self.language == "SELFIES":
            self.text_tokenizer = SelfiesTokenizer()
        elif language == "AAS":
            self.text_tokenizer = CharacterTokenizer()
        elif language == "REACTION_SMILES":
            self.text_tokenizer = ReactionSmilesTokenizer(
                precursor_separator=precursor_separator
            )
            self.presep = precursor_separator
        elif language == "REACTION_SELFIES":
            self.text_tokenizer = ReactionSelfiesTokenizer(
                precursor_separator=precursor_separator
            )
            self.presep = precursor_separator
        elif language == "Polymer":
            self.text_tokenizer = PolymerGraphTokenizer()
        else:
            raise ValueError(
                f"Unsupported language {language}, choose 'SMILES', 'SELFIES', 'AAS', 'REACTION_SMILES', 'REACTION_SELFIES' or 'Polymer'"
            )

        self.property_tokenizer = PropertyTokenizer()
        self.expression_separator = "|"
        self.separator_idx = self.vocab[self.expression_separator]
        self.pad_even = pad_even

        # DEPRECATED
        if pad_even:
            self.pad_even_fn = lambda x: x if len(x) % 2 == 0 else x + [self.pad_token]
        else:
            self.pad_even_fn = lambda x: x

    @property
    def vocab_list(self) -> List[str]:
        """List vocabulary tokens.

        Returns:
            a list of vocabulary tokens.
        """
        return list(self.vocab.keys())

    def update_vocab(self, tokens: Set[str]) -> None:
        """Update the vocabulary with the added tokens.

        Args:
            tokens: tokens that should be added to the vocabulary.
        """

        # Update token to idx mapping and reset added_tokens_encoder
        self.vocab.update(self.added_tokens_encoder)
        self.added_tokens_encoder = {}

        # Update idx to token mapping
        self.ids_to_tokens.update(self.added_tokens_decoder)
        self.added_tokens_decoder = {}

        # Fix a problem in base tokenizer that prevents splitting added tokens:
        # https://github.com/huggingface/transformers/issues/7549
        self.unique_no_split_tokens = list(
            set(self.unique_no_split_tokens).difference(tokens)
        )
        self._create_trie(self.unique_no_split_tokens)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a text representing an expression.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        splitted_expression = text.split(self.expression_separator)
        tokens = []
        for property_expression in splitted_expression[:-1]:
            tokens.extend(self.property_tokenizer.tokenize(property_expression))
            tokens.append(self.expression_separator)
        tokens.extend(self.text_tokenizer.tokenize(splitted_expression[-1]))
        # TODO: remove this hack
        # This is a hack to get around DataCollatorForLanguageModeling requiring even
        # length sequences
        return self.pad_even_fn(tokens)

    def add_padding_tokens(
        self, token_ids: List[int], max_length: int, padding_right: bool = True
    ) -> List[int]:
        """Adds padding tokens to return a sequence of length max_length.

        By default padding tokens are added to the right of the sequence.

        Args:
            token_ids: token indexes.
            max_length: maximum length of the sequence.
            padding_right: whether the sequence is padded on the right. Defaults to True.

        Returns:
            padded sequence of token indexes.
        """
        padding_ids = [self.pad_token_id] * (max_length - len(token_ids))
        if padding_right:
            return token_ids + padding_ids
        else:
            return padding_ids + token_ids

    @staticmethod
    def get_sample_label(mlm_label: List[str], mlm_input: List[str]) -> List[str]:
        """MLM case: Retrieve true sample sequence from mlm label and mlm input.
        NOTE: Also works for PLM.

        Args:
            mlm_label (List[str]): Target sample used in MLM.
            mlm_input (List[str]): MLM input sample.

        Returns:
            List[str]: Sample sequence as part of the dataset
        """

        return [i if el == "[UNK]" else el for el, i in zip(mlm_label, mlm_input)]

    @staticmethod
    def get_sample_prediction(
        mlm_prediction: List[str], mlm_input: List[str]
    ) -> List[str]:
        """MLM case: Retrieve predicted sequence from mlm prediction and mlm input
        NOTE: Also works for PLM.

        Args:
            mlm_label (List[str]): Target sample used in MLM.
            mlm_input (List[str]): MLM input sample.

        Returns:
            List[str]: Sample sequence as part of the dataset
        """
        return [
            i if i not in ["[MASK]"] else o for o, i in zip(mlm_prediction, mlm_input)
        ]

    @staticmethod
    def floating_tokens_to_float(token_ids: List[str]) -> float:
        """Converts tokens representing a float value into a float.
        NOTE: Expects that non-floating tokens are strippped off

        Args:
            token_ids (List[str]): List of tokens, each representing a float.
                E.g.: ['_0_0_', '_._', '_9_-1_', '_3_-2_', '_1_-3_']

        Returns:
            float: Float representation for the list of tokens.
        """
        try:
            float_string = "".join([token.split("_")[1] for token in token_ids])
            float_value = float(float_string)
        except ValueError:
            float_value = -1
        return float_value

    def aggregate_tokens(
        self, token_ids: List[str], label_mode: bool, cls_first: bool = True
    ) -> Tuple[str, Dict]:
        """Receives tokens of one sample and returns sequence (e.g. SMILES) and
        a dict of properties.

        Args:
            token_ids (List[str]): List of tokens.
            label_mode (bool): Whether the token_ids are labels or predictions.
            cls_first (bool, optional): Whether CLS  token occurres first, default: True

        Returns:
            Tuple[str, Dict]:
                str: SMILES/SELFIES sequence of sample.
                Dict: A dictionary with property names (e.g. 'qed') as key and
                    properties as values.
        """
        edx = min(
            token_ids.index("[SEP]") if "[SEP]" in token_ids else 1000,
            token_ids.index("[PAD]") if "[PAD]" in token_ids else 1000,
        )

        edx = -1 if edx == 1000 else edx

        # Special handling for reaction models
        if "REACTION" in self.language:
            en_idxs = [i for i, x in enumerate(token_ids) if x == self.presep]
            for i, idx in enumerate(en_idxs):
                if idx + 2 in en_idxs and token_ids[idx + 1] != ".":
                    logger.info(f"Replacing, {token_ids[idx + 1]} with `.` ")
                    token_ids[idx + 1] = "."

        seq = (
            "".join(token_ids[token_ids.index("|") + 1 : edx])
            if "|" in token_ids
            else "".join(token_ids)
        )
        property_dict = {}
        for idx, t in enumerate(token_ids):
            if t.startswith("<") and t.endswith(">"):
                if "REACTION" in self.language and t == self.presep:
                    continue
                key = t[1:-1]

                # Convert float
                end_floating_idx = idx + 1
                while token_ids[end_floating_idx].startswith("_"):
                    end_floating_idx += 1

                prop = self.floating_tokens_to_float(
                    token_ids[idx + 1 : end_floating_idx]
                )

                property_dict[key] = prop

        return seq, property_dict

    def to_readable(self, sequence: str) -> str:
        """Safely returns a readable string irrespective of whether the language is
        SMILES, SELFIES or AAS.

        Args:
            sequence (str): A string representing a molecule (either SMILES or SELFIES)
                or amino acid sequence.

        Returns:
            str: A SMILES representing the same molecule.
        """
        if self.language == "SMILES":
            return sequence
        elif self.language == "SELFIES":
            return decoder(sequence)
        elif self.language == "AAS":
            return sequence
        elif self.language == "REACTION_SMILES":
            return sequence
        elif self.language == "REACTION_SELFIES":
            try:
                pre, prod = sequence.split(">>")
            except ValueError:
                return ""
            # Replace the entity separators
            pre = pre.replace(self.presep, ".")
            prod = prod.replace(self.presep, ".")
            pres = [decoder(p) for p in pre.split(".")]
            prods = [decoder(p) for p in prod.split(".")]
            # remove empty entities
            pres = [p for p in pres if p]
            prods = [p for p in prods if p]
            return ".".join(pres) + ">>" + ".".join(prods)
        elif self.language == "Polymer":
            return sequence
        else:
            raise AttributeError(f"Unknown language {self.language}")


class InferenceBertTokenizer(ExpressionBertTokenizer):
    """
    InferenceBertTokenizer that implements some additional functionalities and
    sanity check compared to the ExpressionBertTokenizer.

    The primary justification for this class is the necessity to tokenize samples that
    include the [MASK] token in the sequence.
    """

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a text representing an expression.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        tokens = []
        if text.startswith(self.expression_separator):
            text = text[1:]
            tokens.append(self.expression_separator)
        if text.startswith("<") and text.endswith(">"):
            prop_tokens = re.compile(r"\s*(<\w+>)\s").split(text)

            if len(prop_tokens) != 1:
                raise ValueError(f"Problem in processing {text}: ({prop_tokens})")
            tokens.extend(prop_tokens)
            return tokens
        tokens.extend(super()._tokenize(text))
        return tokens

    def tokenize(self, *args, **kwargs) -> List[str]:
        """
        Overwriting parent class method to ensure an even number of tokens *even if
        the input contains masked tokens*.

        Returns:
            A list of tokens
        """
        text = super().tokenize(*args, **kwargs)
        text = text if len(text) % 2 == 0 else text + [self.pad_token]
        return text
        return text
