"""Tokenization utilties for exrepssions."""

import os
import re
from typing import List, Optional, Union

from tokenizers import Tokenizer
from transformers import AlbertTokenizer, BertTokenizer, T5Tokenizer


MAX_ORDER = 15
MIN_ORDER = -12
NUMERICAL_TOKENS = ["_._"] + [
    f"_{d}_{o}_" for o in range(MIN_ORDER, MAX_ORDER) for d in range(10)
]
SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

AA_SEQUENCE_TOKENIZERS_LOADERS = {
    "albert": lambda tokenizer_filepath: AlbertTokenizer.from_pretrained(
        tokenizer_filepath, do_lower_case=False
    ),
    "bert": lambda tokenizer_filepath: BertTokenizer.from_pretrained(
        tokenizer_filepath, do_lower_case=False
    ),
    "generic": lambda tokenizer_filepath: Tokenizer.from_file(tokenizer_filepath),
}

AA_SEQUENCE_TOKENIZER_FUNCTIONS = {
    "albert": lambda text, tokenizer: tokenizer.tokenize(" ".join(list(text))),
    "bert": lambda text, tokenizer: tokenizer.tokenize(" ".join(list(text))),
    "generic": lambda text, tokenizer: tokenizer.encode(text).tokens,
}


class TokenizationError(ValueError):
    def __init__(self, title: str, detail: str):
        """
        Initialize TokenizationError.
        Args:
            title (str): title of the error.
            detail (str): decscription of the error.
        """
        self.type = "TokenizationError"
        self.title = title
        self.detail = detail


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class RegexTokenizer:
    """Run regex tokenization"""

    def __init__(self, regex_pattern: str, suffix: str = "") -> None:
        """Constructs a RegexTokenizer.
        Args:
            regex_pattern: regex pattern used for tokenization.
            suffix: optional suffix for the tokens. Defaults to "".
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)
        self.suffix = suffix

    def tokenize(self, text: str) -> List[str]:
        """Regex tokenization.
        Args:
            text: text to tokenize.
        Returns:
            extracted tokens.
        """

        if len(text) <= 6 or all(
            [
                w not in text
                for w in [
                    "Br",
                    "Cl",
                    "N",
                    "O",
                    "S",
                    "P",
                    "F",
                    "I",
                    "b",
                    "c",
                    "n",
                    "o",
                    "s",
                    "p",
                ]
            ]
        ):
            raise TokenizationError(
                "SmilesMinInput",
                f'Text="{text}"',
            )

        tokens = [token for token in self.regex.findall(text)]

        if text != "".join(tokens):
            raise TokenizationError(
                "SmilesJoinedTokensMismatch",
                f'Text="{text}" != joined_tokens="{"".join(tokens)}"',
            )

        return [f"{token}{self.suffix}" for token in tokens]


class AASequenceTokenizer:
    """Run AA sequence tokenization."""

    def __init__(
        self, tokenizer_filepath: str, tokenizer_type: str = "generic"
    ) -> None:
        """
        Constructs an AASequenceTokenizer.
        Args:
            tokenizer_filepath: path to a serialized AA sequence tokenizer.
            tokenizer_type: type of tokenization to use. Defaults to "generic".
        """
        self.tokenizer_filepath = tokenizer_filepath
        self.tokenizer_type = tokenizer_type
        self.tokenizer = AA_SEQUENCE_TOKENIZERS_LOADERS.get(
            self.tokenizer_type, AA_SEQUENCE_TOKENIZERS_LOADERS["generic"]
        )(self.tokenizer_filepath)
        self.tokenizer_fn = AA_SEQUENCE_TOKENIZER_FUNCTIONS.get(
            self.tokenizer_type, AA_SEQUENCE_TOKENIZER_FUNCTIONS["generic"]
        )

    def tokenize(self, text: str) -> List[str]:
        """Tokenization of a property.
        Args:
            text: text to tokenize.
        Returns:
            extracted tokens.
        """

        if len(text) < 50:
            raise TokenizationError(
                "AASMinInput",
                f'Text="{text}"',
            )

        tokens = self.tokenizer_fn(text, self.tokenizer)

        if text != "".join(tokens):
            raise TokenizationError(
                "AASJoinedTokensMismatch",
                f'Text="{text}" != joined_tokens="{"".join(tokens)}"',
            )

        return tokens


class EnzymaticReactionTokenizer:
    """Constructs a EnzymaticReactionTokenizer using AA sequence."""

    def __init__(
        self,
        aa_sequence_tokenizer_filepath: Optional[str] = None,
        smiles_aa_sequence_separator: str = "|",
        reaction_separator: str = ">>",
        aa_sequence_tokenizer_type: str = "generic",
        smiles_suffix: str = "_",
    ) -> None:
        """Constructs an EnzymaticReactionTokenizer.
        Args:
            aa_sequence_tokenizer_filepath: file to a serialized AA sequence tokenizer.
            smiles_aa_sequence_separator: separator between reactants and AA sequence. Defaults to "|".
            reaction_separator: reaction sides separator. Defaults to ">>".
            aa_sequence_tokenizer_type: type of tokenization to use for aa sequences. Defaults to "generic".
        """
        # define tokenization utilities
        self.smiles_tokenizer = RegexTokenizer(
            regex_pattern=SMILES_TOKENIZER_PATTERN, suffix=smiles_suffix
        )
        self.aa_sequence_tokenizer_filepath = aa_sequence_tokenizer_filepath
        self.aa_sequence_tokenizer_type = aa_sequence_tokenizer_type

        self.aa_sequence_tokenizer = AASequenceTokenizer(
            tokenizer_filepath=self.aa_sequence_tokenizer_filepath,
            tokenizer_type=self.aa_sequence_tokenizer_type,
        )

        self.smiles_aa_sequence_separator = smiles_aa_sequence_separator
        self.reaction_separator = reaction_separator

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a text representing an enzymatic reaction with AA sequence information.
        Args:
            text: text to tokenize.
        Returns:
            extracted tokens.
        """

        if is_number(text):
            raise TokenizationError(
                "EnzymaticReactionTokenizerNumericOnlyInput",
                f'Text="{text}"',
            )

        product = ""
        aa_sequence = ""
        try:
            reactants_and_aa_sequence, product = text.split(self.reaction_separator)
        except ValueError:
            reactants_and_aa_sequence = text
        try:
            reactants_or_aa_sequnce, aa_sequence = reactants_and_aa_sequence.split(
                self.smiles_aa_sequence_separator
            )
        except ValueError:
            reactants_or_aa_sequnce = reactants_and_aa_sequence

        tokens = []

        try:
            tokens.extend(self.smiles_tokenizer.tokenize(reactants_or_aa_sequnce))
        except TokenizationError:
            aa_sequence = reactants_or_aa_sequnce

        if aa_sequence:
            if len(tokens) > 0:
                tokens.append(self.smiles_aa_sequence_separator)
            aa_tokens = [
                f"{token}_"
                for token in self.aa_sequence_tokenizer.tokenize(aa_sequence)
            ]
            tokens.extend(aa_tokens)

        if product:
            tokens.append(self.reaction_separator)
            tokens.extend(self.smiles_tokenizer.tokenize(product))

        return tokens


class MultiFloatTokenizer:
    """A float tokenizer.
    Example:
    >>>tok.tokenize("-12.023")
    [['_-_', '_1_1_', '_2_0_', '_._', '_0_-1_', '_2_-2_', '_3_-3_']]
    >>>tok.tokenize("First take 12.3mL of water and then add to -12.023 g of salt 3 times")
    [['_1_1_', '_2_0_', '_._', '_3_-1_'], ['_-_', '_1_0_', '_._', '_2_-1_', '_3_-2_', '_4_-3_']]
    """

    def __init__(self) -> None:
        """Constructs a PropertyTokenizer."""
        self.regex = re.compile(r".*?(\d+)(\.)?(\d+)?\s*")
        self.separator_token = "->"

    def tokenize(self, text: str) -> List[List[str]]:
        """Tokenization of a property.
        Args:
            text: text to tokenize.
        Returns:
            List of list of extracted property tokens.
        """
        all_tokens = []
        remaining_text = text
        matched = self.regex.match(remaining_text)

        if matched:
            all_tokens = []
            while matched:
                tokens = []
                units, dot, decimals = matched.groups()

                tokens += [
                    f"_{number}_{position}_"
                    for position, number in enumerate(units[::-1])
                ][::-1]
                if dot:
                    tokens += [f"_{dot}_"]
                if decimals:
                    tokens += [
                        f"_{number}_-{position}_"
                        for position, number in enumerate(decimals, 1)
                    ]
                remaining_text = remaining_text[matched.end() :]
                matched = self.regex.match(remaining_text)
                all_tokens.append(tokens)
        return all_tokens


class T5FloatTokenizer(T5Tokenizer):
    def __init__(self, *args, **kwargs):
        self.float_tokenizer = MultiFloatTokenizer()
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        **kwargs,
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *init_inputs, **kwargs
        )

        tokenizer.add_tokens(NUMERICAL_TOKENS)

        return tokenizer

    def tokenize(self, text, **kwargs) -> List[str]:
        float_tokens = self.float_tokenizer.tokenize(text)

        matches = [match for match in self.float_tokenizer.regex.finditer(text)]

        if len(matches) == 0:
            return super()._tokenize(text, **kwargs)

        combined_tokens = []
        current_idx = 0
        float_idx = 0

        for match in matches:
            first_numerical_index = re.search(r"\d", match.group()).start()

            if match.start() + first_numerical_index > current_idx:
                # add a dummy text in the beginning to avoid the introduction of wrong blank tokens
                tokens = super()._tokenize(
                    "@T5Float@"
                    + text[current_idx : match.start() + first_numerical_index],
                    **kwargs,
                )
                combined_tokens.extend(tokens[6:])

            combined_tokens.extend(float_tokens[float_idx])

            current_idx = match.end()
            float_idx += 1

        if current_idx < len(text):
            # add a dummy text in the beginning to avoid the introduction of wrong blank tokens
            tokens = super()._tokenize("@T5Float@" + text[current_idx:], **kwargs)
            combined_tokens.extend(tokens[6:])

        return combined_tokens


class T5SmilesAATokenizer(T5FloatTokenizer):
    def __init__(
        self,
        smiles_vocabulary_path: str,
        aa_tokenizer_filepath: str,
        suffix: str = "_",
        **kwargs,
    ):
        """Constructs a RegexTokenizer."""

        super().__init__(**kwargs)

        smiles_vocabulary = []

        with open(smiles_vocabulary_path, "r") as fp:
            for line in fp:
                smiles_vocabulary.append(line.strip())

        self.enzymatic_tokenizer = EnzymaticReactionTokenizer(
            aa_sequence_tokenizer_filepath=aa_tokenizer_filepath, smiles_suffix=suffix
        )

        self.number_of_utilized_text_tokens = len(self)
        self.add_tokens(smiles_vocabulary)
        aas_vocabulary = [
            f"{token}_"
            for token in self.enzymatic_tokenizer.aa_sequence_tokenizer.tokenizer.get_vocab().keys()
            if not (token.startswith("<") and token.endswith(">"))
        ]
        self.add_tokens(aas_vocabulary)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a given text.
        Args:
            text: text to be tokenized.
        Returns:
            tokens
        """

        text = text.split()

        tokenized_text = []

        for i, word in enumerate(text):
            try:

                tokenized_aas = self.enzymatic_tokenizer.tokenize(word)
                tokenized_text.append("▁")
                tokenized_text.extend(tokenized_aas)

            except TokenizationError:

                tokenized_str = super().tokenize(word)

                if not tokenized_str[0].startswith("▁"):
                    tokenized_text.append("▁")
                tokenized_text.extend(tokenized_str)

        return tokenized_text

    def clean_up_token(self, token: str) -> str:
        """Clean up token from special characters.
        Args:
            token: input token.
        Returns:
            cleaned token
        """

        if token.endswith("_") and len(token) > 1:
            if token.startswith("_"):
                return token[1]
            return token[:-1]

        return token

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens to string.
        Args:
            text: list of tokens.
        Returns:
            string
        """

        tokens = [self.clean_up_token(token) for token in tokens]
        return super().convert_tokens_to_string(tokens)
