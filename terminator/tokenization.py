"""Tokenization utilties for exrepssions."""
import re
from typing import Dict, List, Tuple

import torch
from selfies import decoder, split_selfies
from transformers import BertTokenizer, XLNetTokenizer

SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"


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


class PropertyTokenizerSquare(PropertyTokenizer):
    """Run a property tokenization."""

    def __init__(self) -> None:
        """Constructs a PropertyTokenizer."""
        self.regex = re.compile(r"\s*(\[\w+\])\s*?(\+|-)?(\d+)(\.)?(\d+)?\s*")


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
        self.tokenizer = lambda x: list(split_selfies(x))


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
        else:
            raise ValueError(
                f"Unsupported language {language}, choose 'SMILES', 'SELFIES' or 'AAS'."
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
    Constructs a bert-based tokenizer used for the Regression Transformer.

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
                'SMILES', 'SELFIES' or 'AAS'.
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
        else:
            raise ValueError(
                f"Unsupported language {language}, choose 'SMILES', 'SELFIES' or 'AAS'."
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
            token_ids: List of tokens, each representing a float.
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

        seq = (
            "".join(token_ids[token_ids.index("|") + 1 : edx])
            if "|" in token_ids
            else "".join(token_ids)
        )
        property_dict = {}
        for idx, t in enumerate(token_ids):
            if t.startswith("<") and t.endswith(">"):
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
        else:
            raise AttributeError(f"Unknown language {self.language}")


class XLNetRTTokenizer(XLNetTokenizer):
    """
    A XLNet-based tokenizer for the Regression Transformer, build for the
    humicroedit dataset
    """

    def set_property_tokenizer(
        self,
        tokenizer: PropertyTokenizer,
        expression_separator: str = "{",
        expression_end: str = "}",
        property_token: str = "[funny]",
    ):
        """
        Set the property tokenizer to be used by the main tokenizer.

        Args:
            tokenizer: a property tokenizer.
            expression_separator: a token that separates the property from the rest.
            expression_end: a token that ends the joke-token sequence.
            property_token: the property token.
        """
        self.property_tokenizer = tokenizer
        # The start token indicating the joke tokens
        self.expression_separator = expression_separator
        self.expressiond_end = expression_end
        self.property_token = property_token

    def set_vocab(self):
        self.vocab = self.get_vocab()
        self.idx_to_token = dict(zip(self.vocab.values(), self.vocab.keys()))

    def _tokenize(self, text: str) -> List[str]:
        """
        Core tokenization function.

        Args:
            text: A string to be tokenized.

        Returns:
            A list of tokens.
        """
        prop, rest = text.split(self.expression_separator)
        tokens = self.property_tokenizer.tokenize(prop) + [self.expression_separator]

        entities = rest.split(self.expressiond_end)
        for idx, entity in enumerate(entities):
            tokens.extend(super()._tokenize(entity))
            if idx < len(entities) - 1:
                tokens.extend([self.expressiond_end])

        if len(tokens) % 2 != 0:
            tokens + [self.pad_token]
        return tokens

    @property
    def vocab_list(self):
        return list(self.vocab.keys())

    @staticmethod
    def floating_tokens_to_float(token_ids: List[str]) -> float:
        """Converts tokens representing a float value into a float.
        NOTE: Expects that non-floating tokens are strippped off

        Args:
            token_ids: List of tokens, each representing a float.
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

    def decode_internal(self, token_ids: torch.Tensor, *args, **kwargs) -> str:
        tokens = ""
        for _id in token_ids.tolist():
            token = self.idx_to_token[_id] if _id != -100 else "[UNK]"
            tokens += token + " "
        return tokens

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
            i if i not in ["[MASK]", "<mask>"] else o
            for o, i in zip(mlm_prediction, mlm_input)
        ]

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

        seq = (
            "".join(token_ids[token_ids.index("|") + 1 : edx])
            if "|" in token_ids
            else "".join(token_ids)
        )
        property_dict = {}
        for idx, t in enumerate(token_ids):
            if t == self.property_token:
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
