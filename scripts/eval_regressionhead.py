#!/usr/bin/env python3
"""
Language modeling adapted from Huggingface transformers.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from scipy.stats import pearsonr, spearmanr
from selfies import decoder, encoder
from sklearn.metrics import mean_squared_error
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
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
    XLNetForSequenceClassification,
    XLNetLMHeadModel,
    get_linear_schedule_with_warmup,
    set_seed,
)
from transformers.tokenization_utils_base import BatchEncoding

from terminator.args import CustomTrainingArguments, ModelArguments
from terminator.collators import TRAIN_COLLATORS
from terminator.datasets import get_dataset
from terminator.tokenization import ExpressionBertTokenizer
from terminator.trainer import CustomTrainer, get_trainer_dict
from terminator.utils import get_latest_checkpoint

transformers.logging.set_verbosity_info()
logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.DEBUG)


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
    batch_size: Optional[int] = field(default=16, metadata={"help": "Batch size"})


class XLNetRegressionDataset(Dataset):
    def __init__(self, tokenizer, data_path):

        self.tokenizer = tokenizer

        # Lazy data loading
        with open(data_path, "r") as f:
            self.examples = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        prop, molecules = self.examples[i].split("|")
        label = float(prop.split(">")[-1])
        model_input = self.tokenizer(molecules)
        return model_input, label


@dataclass
class Collator(DataCollatorForPermutationLanguageModeling):
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
        self, examples: List[Tuple[Dict[str, List[int]], float]]
    ) -> Dict[str, torch.Tensor]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_inputs = [e[0]["input_ids"] for e in examples]
        inputs = self._tensorize_batch(model_inputs)
        inputs = self.finalize(inputs)

        attention_mask = self.attention_mask(inputs)

        labels = torch.Tensor([e[-1] for e in examples])
        return labels.to(device), {
            "input_ids": inputs.to(device),
            "attention_mask": attention_mask.to(device),
        }


def main():

    # Switch off comet
    os.environ["COMET_MODE"] = "DISABLED"

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    print(model_args)
    print(data_args)

    if not os.path.exists(train_args.output_dir):
        raise ValueError(f"Output directory ({train_args.output_dir}) does not exists!")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if train_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        train_args.local_rank,
        train_args.device,
        train_args.n_gpu,
        bool(train_args.local_rank != -1),
        train_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", train_args)

    # Set seed
    set_seed(train_args.seed)

    output_dir = train_args.output_dir
    model = XLNetForSequenceClassification.from_pretrained(
        output_dir,
        cache_dir=model_args.cache_dir,
        mem_len=1024,
        return_dict=True,
    )

    logger.info(f"Model restored from {output_dir}")

    tokenizer = ExpressionBertTokenizer.from_pretrained(model_args.tokenizer_name)

    logger.info(f"PyTorch version: {torch.__version__}")
    # model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fileprefix = data_args.eval_data_file.split("/")[-1].split(".")[0]
    logger.info(f"Results will be saved in {output_dir} with prefix {fileprefix}")
    # WHY ARE THE CORRELATIONS NEGATIVE? YEST WITH VALIDATIAON DATA
    dataset = XLNetRegressionDataset(
        tokenizer=tokenizer, data_path=data_args.eval_data_file
    )
    model = model.to(device)
    collator = Collator(tokenizer=tokenizer)
    logger.info(f"Evaluation dataset size: {len(dataset)}.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"{total_params} parameters, {num_params} trainable. Model: {type(model)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=data_args.batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=collator,
    )

    eval_seqs = [dataset.examples[i].split("|")[-1] for i in range(len(dataset))]

    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for idx, (labs, inputs) in enumerate(loader):
            output = model(**inputs, labels=labs)
            prediction = output.logits.cpu().detach().squeeze().numpy()

            labels.extend(list(labs.cpu().detach().numpy()))
            predictions.extend(list(prediction))

    rmse = np.sqrt(mean_squared_error(predictions, labels))
    pearson = pearsonr(predictions, labels)[0]
    spearman = spearmanr(predictions, labels)[0]

    logger.info(
        f"Eval: RMSE:{rmse:.5f}, pearson:{pearson:.5f}, spearman:{spearman:.5f}"
    )

    with open(os.path.join(output_dir, f"{fileprefix}_results.json"), "w") as f:
        json.dump(
            {"RMSE": str(rmse), "Pearson": str(pearson), "Spearman": str(spearman)},
            f,
            indent=4,
        )
    pd.DataFrame(
        {
            "sequence": eval_seqs,
            "predictions": list(predictions),
            "labels": list(labels),
        }
    ).to_csv(os.path.join(output_dir, f"{fileprefix}_predictions.csv"))


if __name__ == "__main__":
    main()
