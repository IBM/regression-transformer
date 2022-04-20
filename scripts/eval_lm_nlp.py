#!/usr/bin/env python3
"""
Language modeling adapted from Huggingface transformers.

The file is an adaptation of https://github.com/huggingface/transformers/blob/v3.1.0/examples/language-modeling/run_language_modeling.py

"""

import json
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    XLNetConfig,
    XLNetLMHeadModel,
    set_seed,
)

from terminator.args import CustomTrainingArguments, EvalArguments, ModelArguments
from terminator.collators import TRAIN_COLLATORS, PropertyCollator
from terminator.datasets import get_dataset
from terminator.evaluator import Evaluator
from terminator.tokenization import PropertyTokenizerSquare, XLNetRTTokenizer
from terminator.trainer import CustomTrainer, get_trainer_dict
from terminator.utils import get_latest_checkpoint

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    eval_data_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )

    mlm: bool = field(
        default=False,
        metadata={
            "help": "Train with masked-language modeling loss instead of language modeling."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5,
        metadata={
            "help": "Maximum length of a span of masked tokens for permutation language modeling."
        },
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((CustomTrainingArguments, EvalArguments))
    training_args, eval_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    with open(eval_args.param_path, "r") as f:
        eval_params = json.load(f)

    # Wrap into args to be safe
    eval_args.__dict__.update(eval_params)

    if not os.path.exists(training_args.output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) does not exist"
        )
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Set seed
    set_seed(training_args.seed)

    model_dir = training_args.output_dir
    if "checkpoint" not in model_dir:
        model_dir = get_latest_checkpoint(
            model_dir, must_contain=eval_params.get("checkpoint-str", "best")
        )

    config_name = os.path.join(model_dir, "config.json")
    with open(config_name, "r") as f:
        model_params = json.load(f)

    config = AutoConfig.from_pretrained(config_name)

    model = XLNetLMHeadModel.from_pretrained(model_dir, config=config)
    logger.info(f"Model restored from {model_dir}")

    tokenizer = XLNetRTTokenizer.from_pretrained(model_dir)
    property_tokenizer = PropertyTokenizerSquare()
    tokenizer.set_property_tokenizer(property_tokenizer)
    tokenizer.set_vocab()
    # Otherwise the freshly added tokens are added as special tokens.
    # tokenizer.unique_no_split_tokens = tokenizer.unique_no_split_tokens[:9]

    logger.info(f"PyTorch version: {torch.__version__}")
    # model.resize_token_embeddings(len(tokenizer))

    if eval_params.get("block_size", -1) <= 0:
        eval_params["block_size"] = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        eval_params["block_size"] = min(training_args.block_size, tokenizer.max_len)

    eval_dataset = get_dataset(
        eval_args.eval_file,
        tokenizer=tokenizer,
        block_size=eval_params["block_size"],
        line_by_line=eval_params.get("line_by_line", True),
    )

    logger.info(f"Dataset sizes, {len(eval_dataset)}.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters {num_params} of type {type(model)}")

    custom_trainer_params = get_trainer_dict(model_params)

    _unused_collator = DataCollatorForPermutationLanguageModeling(
        tokenizer=tokenizer, plm_probability=0.1, max_span_length=2
    )

    # Initialize our Evaluator
    evaluator = Evaluator(
        model=model,
        args=training_args,
        eval_params=eval_params,
        data_collator=_unused_collator,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        prediction_loss_only=False,
        **custom_trainer_params,
    )

    # Evaluation
    result_dir = os.path.join(model_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    eval_filename = eval_args.eval_file.split("/")[-1].split("_")[-1].split(".")[0]
    logger.info("*** Evaluate perplexity ***")

    property_results = []
    properties = eval_params["property_token"]
    orders = eval_params.get("property_token_masking_order", None)
    tokens_to_mask = eval_params.get("property_tokens_to_mask", None)

    for prop, order, mask in zip(properties, orders, tokens_to_mask):
        logger.info(f"*** Evaluate property {prop} ***")

        for to_mask in mask:

            # We iteratively make the task harder by masking 1-4 tokens.
            # The order of this is determined by `property_token_masking_order`.
            property_collator = PropertyCollator(
                tokenizer=tokenizer,
                property_tokens=[prop],
                num_tokens_to_mask=[to_mask],
                mask_token_order=[order],
            )
            print(f"Masking {to_mask} in order {order}")
            ps, rs, ss = evaluator.property_prediction(
                property_collator,
                save_path=os.path.join(
                    result_dir, f"{prop[1:-1]}_{eval_filename}_mask_{to_mask}.csv"
                ),
            )
            for p, r, s, n in zip(ps, rs, ss, ["Greedy", "Sampling", "Beam"]):
                prop_res_dict = {
                    "prop": prop[1:-1],
                    "pearson": p,
                    "spearman": s,
                    "rmse": r,
                    "search": n,
                    "num_masked": to_mask,
                }
                property_results.append(prop_res_dict)

        pd.DataFrame(property_results).to_csv(
            os.path.join(result_dir, f"property_prediction_{eval_filename}.csv")
        )


if __name__ == "__main__":
    main()
