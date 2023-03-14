"""Create a vocabulary."""
import argparse
import os
from collections import Counter

from tqdm import tqdm

from terminator.tokenization import ExpressionTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "input_filepath", type=str, help="data used to create a vocabulary."
)
parser.add_argument(
    "output_filepath", type=str, help="output where to store the vocabulary."
)
parser.add_argument(
    "--max_exponent", type=int, default=5, help="maximum exponent for num-tokens."
)


def main() -> None:
    """Create a vocabulary using an ExpressionTokenizer."""
    args = parser.parse_args()
    input_filepath = args.input_filepath
    output_filepath = args.output_filepath
    max_exponent = args.max_exponent

    vocabulary_counter = Counter()
    tokenizer = ExpressionTokenizer()

    # tokens for properties
    vocabulary_counter.update(
        [
            "<qed>",
            "<logp>",
            "<molwt>",
            "<sas>",
            "<scs>",
            "<esol>",
            "<plogp>",
            "<lipinski>",
            "<rxnretro>",
            "<aromatic>",
        ]
    )
    # tokens for property numerical values
    digits = list(range(10))
    vocabulary_counter.update(
        [
            f"_{digit}_{exponent}_"
            for exponent in range(max_exponent + 1)
            for digit in digits
        ]
        + [
            f"_{digit}_-{exponent}_"
            for exponent in range(max_exponent + 1)
            for digit in digits
        ]
    )
    with open(input_filepath, "rt") as fp:
        for line in tqdm(fp):
            vocabulary_counter.update(tokenizer.tokenize(line.strip()))

    # special tokens for the model training and keeping the possibility to extend the vocabulart
    special_tokens = [
        "[PAD]",
        "[unused1]",
        "[unused2]",
        "[unused3]",
        "[unused4]",
        "[unused5]",
        "[unused6]",
        "[unused7]",
        "[unused8]",
        "[unused9]",
        "[unused10]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
    ]

    with open(output_filepath, "wt") as fp:
        tokens = special_tokens + [
            token for token, _ in vocabulary_counter.most_common()
        ]
        fp.write(os.linesep.join(tokens))


if __name__ == "__main__":
    main()
