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

from terminator.args import CustomTrainingArguments, ModelArguments
from terminator.collators import TRAIN_COLLATORS
from terminator.datasets import get_dataset
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

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
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

    # Switch off comet
    os.environ["COMET_MODE"] = "DISABLED"

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    configuration = XLNetConfig("xlnet-base-cased")
    config_dict = configuration.to_dict()

    model = XLNetLMHeadModel.from_pretrained(
        "xlnet-base-cased",
        cache_dir=model_args.cache_dir,
        mem_len=1024,
        return_dict=True,
    )

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    tokenizer = XLNetRTTokenizer.from_pretrained(
        model_args.tokenizer_name, cache_dir=model_args.cache_dir
    )
    property_tokenizer = PropertyTokenizerSquare()
    tokenizer.set_property_tokenizer(property_tokenizer)
    tokenizer.set_vocab()
    # Otherwise the freshly added tokens are added as special tokens.
    tokenizer.unique_no_split_tokens = tokenizer.unique_no_split_tokens[:9]

    if model_args.model_name_or_path:

        # Restore checkpoint if available
        if "checkpoint" not in model_args.model_name_or_path:
            ckpt_path = get_latest_checkpoint(
                model_args.model_name_or_path, must_contain="rmse"
            )
        else:
            ckpt_path = model_args.model_name_or_path

        model = XLNetLMHeadModel.from_pretrained(
            ckpt_path,
            cache_dir=model_args.cache_dir,
            mem_len=1024,
            return_dict=True,
        )
        logger.info(f"Model restored from {ckpt_path}")

        # Get min loss so far
        try:
            loss_df = pd.read_csv(
                os.path.join(ckpt_path, "training_log.csv"),
                index_col=0,
            )
            configuration.update({"training_logs": list(loss_df.T.to_dict().values())})
            logger.info("Restored training loss history.")
        except Exception:
            logger.warning(
                "Could not find loss history, might overwrite good checkpoints."
            )

    logger.info(f"PyTorch version: {torch.__version__}")
    model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    train_dataset = get_dataset(
        data_args.train_data_file,
        tokenizer=tokenizer,
        block_size=data_args.block_size,
    )

    eval_dataset = get_dataset(
        data_args.eval_data_file,
        tokenizer=tokenizer,
        block_size=data_args.block_size,
        line_by_line=data_args.line_by_line,
    )

    logger.info(f"Dataset sizes {len(train_dataset)}, {len(eval_dataset)}.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters {num_params} of type {type(model)}")

    # Set up the training strategy (PLM vs. alternating tasks) + loss function
    if training_args.training_config_path is not None:
        with open(training_args.training_config_path, "r") as f:
            train_config = json.load(f)

        # Store training config file in model directory
        with open(
            os.path.join(training_args.output_dir, "training_configs.json"), "w"
        ) as f:
            json.dump(train_config, f, indent="\t")
    else:
        train_config = {}

    if train_config.get("alternate_tasks", False):
        logger.info("Training with alternate tasks")
        # The main collator is the one for property prediction
        data_collator = TRAIN_COLLATORS["property"](
            tokenizer=tokenizer,
            property_tokens=train_config["property_tokens"],
            num_tokens_to_mask=train_config.get("num_tokens_to_mask", None),
            mask_token_order=train_config.get("mask_token_order", None),
        )
        alternating_collator = TRAIN_COLLATORS[train_config["cg_collator"]](
            tokenizer=tokenizer, **train_config["cg_collator_params"]
        )

    else:
        logger.info("Training with PLM")
        # Only vanilla PLM training
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=data_args.plm_probability,
            max_span_length=data_args.max_span_length,
        )
        alternating_collator = None

    custom_trainer_params = get_trainer_dict(config_dict)

    # Initialize our Trainer
    print("***DATA COLLATOR", data_collator)
    print("***ALTERNATING COLLATOR", alternating_collator)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        prediction_loss_only=False,
        alternating_collator=alternating_collator,
        train_config=train_config,
        **custom_trainer_params,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None
            and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


if __name__ == "__main__":
    main()
