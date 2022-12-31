"""
Taken from https://github.com/huggingface/transformers/blob/v3.1.0/src/transformers/trainer.py
"""

import collections
import gc
import json
import os
import shutil
import warnings
from random import random
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers import Trainer, XLNetTokenizer
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    set_seed,
)
from transformers.utils import logging

from terminator.collators import TRAIN_COLLATORS
from terminator.factories import MODEL_TO_EMBEDDING_FN, NUM_ENCODING_FACTORY
from terminator.search import SEARCH_FACTORY
from terminator.trainer_utils import (
    DistributedTensorGatherer,
    distributed_concat,
    get_trainer_dict,
    nested_concat,
    nested_numpify,
)

logger = logging.get_logger(__name__)

NON_MODEL_KEYS = ["real_property", "sample_weights"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):

        # logger.info(f'ARGS are\n{kwargs}\n{args}')
        # Remove keyword arguments unwanted by parent class
        child_kwargs = get_trainer_dict(kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k not in child_kwargs}

        # Call parent class constructor
        super().__init__(*args, **kwargs)

        # Extract custom arguments
        self.verbose_evaluation = child_kwargs.get("verbose_evaluation", True)
        logger.info(f"Verbose evaluation {self.verbose_evaluation}")

        # Restore the logged parameters (training)
        self.logs = child_kwargs.get("training_logs", [])
        self.eval_logs = []

        # Will safe RMSE and Pearson of every epoch
        try:
            tokens = self.data_collator.property_tokens
        except AttributeError:
            tokens = [None]

        self.perfs = np.tile(
            np.expand_dims(np.vstack([10 * np.arange(3)] * len(tokens)), -1),
            10000,
        ).astype(float)
        self.cidx = 0
        self.cg_perfs = -1 * np.ones((2, 10000))

        if self.logs != []:
            self.min_loss = pd.DataFrame(self.logs)["loss"].min()
            if child_kwargs.get("train_config", {}).get("reset_training_loss", False):
                self.min_loss = 10e5
            logger.info(f"Current minimal loss {self.min_loss}")
        else:
            self.min_loss = 10e5

        self.use_numerical_encodings = child_kwargs.get(
            "use_numerical_encodings", False
        )

        if self.use_numerical_encodings:
            logger.info("Attempting to use numerical encodings.")
            self.numerical_encodings_type = child_kwargs.get(
                "numerical_encodings_type", "float"
            )
            self.numerical_encodings_format = child_kwargs.get(
                "numerical_encodings_format", "sum"
            )
            self.numerical_encodings_dim = child_kwargs.get(
                "numerical_encodings_dim", 16
            )

            if self.numerical_encodings_format == "concat":

                if self.numerical_encodings_dim > child_kwargs["d_model"]:
                    raise ValueError(
                        "Numerical encoding size cant be bigger than embedding size"
                    )

                self.combine_embed = self.overwrite_embed

            elif self.numerical_encodings_format == "sum":
                self.numerical_encodings_dim = child_kwargs["d_model"]

                self.combine_embed = self.sum_embed

            else:
                raise ValueError(
                    f"Unknown float encoding format {self.numerical_encodings_format}."
                )

            self.numerical_encoder = NUM_ENCODING_FACTORY[
                self.numerical_encodings_type
            ](
                num_embeddings=child_kwargs["vocab_size"],
                embedding_dim=self.numerical_encodings_dim,
                vocab=self.tokenizer.vocab,
                vmax=child_kwargs.get("vmax", None),
            )

            self.model_embed = eval(
                MODEL_TO_EMBEDDING_FN[child_kwargs.get("model_type", "xlnet")]
            )

        self.search = SEARCH_FACTORY[child_kwargs.get("eval_search", "greedy")](
            child_kwargs.get("eval_search_args", {})
        )

        self.alternating_collator = child_kwargs.get("alternating_collator", None)
        self.alt_training = self.alternating_collator is not None
        self.save_attention = child_kwargs.get("save_attention", False)

        # Whether we train regular PLM or alternating (PP vs. CD)
        if self.alt_training:

            self.train_config = child_kwargs["train_config"]
            self.cg_mode = False  # Whether we are in PP or in CD mode

            # Set up the routine for alternating training
            self.alt_train_loader = self.get_alt_train_dataloader(
                collator=self.alternating_collator
            )
            if self.eval_dataset is not None:
                self.alt_eval_loader = self.get_alt_eval_dataloader(
                    collator=self.alternating_collator
                )

            # Sanity checks
            self.alternate_steps = self.train_config.get("alternate_steps", 8)
            if (
                self.alternate_steps > (self.args.logging_steps / 2)
                or self.args.logging_steps % self.alternate_steps != 0
                or (self.args.logging_steps / self.alternate_steps) % 2 != 0
            ):
                raise ValueError(
                    f"Combination of alternate steps {self.alternate_steps} and logging"
                    f" steps ({self.args.logging_steps}) would break best-model-saving."
                )
            if (
                self.args.gradient_accumulation_steps > self.alternate_steps
                or self.args.eval_accumulation_steps > self.alternate_steps
                or self.alternate_steps % self.args.gradient_accumulation_steps != 0
                or self.alternate_steps % self.args.eval_accumulation_steps != 0
            ):
                raise ValueError(
                    f"Combination of alternate steps ({self.alternate_steps}) & gradient"
                    f" accumulation steps ({self.args.gradient_accumulation_steps} and "
                    f"{self.args.eval_accumulation_steps}) breaks training logic."
                )

            self.cc_loss_weight = self.train_config.get("cc_loss_weight", 1)
            self.cc_loss = self.train_config.get("cc_loss", False)

            # Implement sample-weighted loss function
            self.cg_bce_loss_fn = CrossEntropyLoss(reduction="none")

            if self.cc_loss:
                self.cc_loss_fn = CrossEntropyLoss(reduction="none")
                # This collator is used in the generation task to predict property/ies
                # of the just generated molecule.
                self.cc_collator = TRAIN_COLLATORS["property"](
                    tokenizer=self.tokenizer,
                    property_tokens=self.alternating_collator.property_tokens,
                    num_tokens_to_mask=[-1]
                    * len(self.alternating_collator.property_tokens),
                    ignore_errors=True,
                )

    def get_alt_train_dataloader(self, collator) -> DataLoader:
        """
        Returns a dataloader for alternating training (conditional generation task)
        NOTE:
        This dataloader is used *in addition* to the regular training dataloader
        handled by the parent class. The intended usage is to use the dataloaders
        in an alternating fashion during training, where this DataLoader is responsible
        for the conditional generation task and the parent class loader is responsible
        for the property prediction task.
        """

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self._get_train_sampler(),
            collate_fn=collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def get_alt_eval_dataloader(self, collator) -> DataLoader:
        """
        Returns a dataloader for alternating evaluation (conditional generation task)
        """

        if self.eval_dataset is None:
            raise ValueError("Evaluation requires an eval_dataset.")

        return DataLoader(
            self.eval_dataset,
            sampler=self._get_eval_sampler(self.eval_dataset),
            batch_size=self.args.eval_batch_size,
            collate_fn=collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def sum_embed(self, e: torch.Tensor, num_e: torch.Tensor) -> torch.Tensor:
        return e + num_e

    def overwrite_embed(self, e: torch.Tensor, num_e: torch.Tensor) -> torch.Tensor:
        e[:, :, -self.numerical_encodings_dim :] = num_e
        return e

    def save_attention(self, inputs: torch.Tensor, attention: torch.Tensor):
        """
        Save the attention weights for the current batch.

        Args:
            inputs (torch.Tensor): input_ids
            attention (torch.Tensor): attention tensor

        """

        for idx, a in enumerate(attention):
            for i, aa in enumerate(a):
                np.save(
                    f"batch_{self.counter}_layer_{idx}_tup_{i}", aa.detach().numpy()
                )

        for i, inp in enumerate(inputs):
            tokens = self.tokenizer.convert_ids_to_tokens(inp.tolist())
            with open(f"batch_{self.counter}_sample_{i}.txt", "w") as f:
                f.write(str(tokens))
        self.counter += 1

    def feed_model(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Forward pass of `inputs` through `model`. This function handles the numerical
        encodings if applicable.

        Args:
            model (nn.Module): The model to consume data.
            inputs (Dict[str, Union[torch.Tensor, Any]]): A dict that can be understood
                by model.__call__. Keys should include `input_ids`, `perm_mask`,
                `labels` and `target_mapping`.

        Returns:
            Dict[str, Union[torch.Tensor, Any]]: Output from model
        """
        model_inputs = inputs  # shallow copy

        if self.use_numerical_encodings:
            model_inputs = inputs.copy()
            # Pop keys unused by model
            [model_inputs.pop(k, None) for k in NON_MODEL_KEYS]
            embeddings = self.model_embed(inputs["input_ids"])
            numerical_embeddings = self.numerical_encoder(inputs["input_ids"])
            embeddings = self.combine_embed(embeddings, numerical_embeddings)
            model_inputs.pop("input_ids", None)

            if not self.save_attention:
                outputs = model(inputs_embeds=embeddings, **model_inputs)
            else:
                # Attention config
                outputs = model(
                    inputs_embeds=embeddings,
                    **model_inputs,
                    output_attentions=True,
                    output_hidden_states=False,
                )
                self.save_attention(inputs["input_ids"], outputs[-1])

        else:
            if self.alt_training and self.cg_mode:
                model_inputs = inputs.copy()
            [model_inputs.pop(k, None) for k in NON_MODEL_KEYS]
            outputs = model(**model_inputs)

        # If we enter here, we are in a training step where we are doing cond. gen.
        if self.alt_training and self.cg_mode:

            # Apply conditional generation loss (BCE loss on affected tokens) with
            # custom sample weights to reflect
            #   1) distance of sampled condition to real prop (vanilla CG collator)
            #   2) "negative learning" (Bimodal generator)
            logits = outputs[1].permute(0, 2, 1)
            sample_loss = self.cg_bce_loss_fn(logits, inputs["labels"]).mean(axis=-1)

            loss = (inputs["sample_weights"] * sample_loss).mean()

            if self.cc_loss:
                # Apply cycle-consistency loss
                cc_loss = self.get_cc_loss(model, inputs, outputs)
                loss += cc_loss * self.cc_loss_weight

            # Overwrite PLM loss with CC loss
            outputs = (loss, *outputs[1:])

        return outputs

    def get_cc_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        outputs: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        """Computes self-consistency loss. Receives the model, model inputs for
            conditional generation and the generated molecules.
            Performs a property prediction on the generated molecules and computes the
            loss between the predicted property of the generated molecule and the
            true property.

        Args:
            model (nn.Module): XLNetLMHeadModel
            inputs (Dict[str, torch.Tensor]): Dict of inputs for the model. Should have
                keys 'input_ids' and 'labels' (at least).
            outputs (Tuple[torch.Tensor]): Outputs from the model for the CD generation
                task. Usually a 3-tuple (loss, logits, mems)

        Returns:
            torch.Tensor: Scalar tensor with CC loss.
        """
        # to avoid recursive cycle since `feed_model` is called below
        self.cg_mode = False

        # Extract logits and plain BCE loss
        _loss, logits = outputs[:2]

        # Convert logits to molecules
        predictions = torch.argmax(logits, dim=-1)

        # Combine predictions with labels
        generations = inputs["input_ids"].clone()
        generations[generations == self.tokenizer.mask_token_id] = predictions[
            generations == self.tokenizer.mask_token_id
        ]

        # mask properties (collator normally works on CPU but in this case we pass device)
        cc_input = self.cc_collator.mask_tokens(generations, device=DEVICE)
        cc_attention_mask = self.cc_collator.attention_mask(generations)
        cc_input = {
            "input_ids": cc_input[0],
            "perm_mask": cc_input[1],
            "target_mapping": cc_input[2],
            "labels": cc_input[3],
            "attention_mask": cc_attention_mask,
        }

        # Pass through model
        cc_outputs = self.feed_model(model, cc_input)
        cc_loss, cc_logits = cc_outputs[:2]
        cc_logits = cc_logits.permute(0, 2, 1)

        # Compute BCE loss between logits and derived labels
        # Reduction is none so the mean reduces from N x T to a scalar.
        loss = self.cc_loss_fn(cc_logits, cc_input["labels"]).mean()

        assert _loss != loss, f"Losses cant be identical: {loss}"
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        NOTE: Overwritten here to enable custom embeddings + for moinitoring purposes.

        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """

        has_labels = any(
            inputs.get(k) is not None
            for k in ["labels", "lm_labels", "masked_lm_labels"]
        )

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            # NOTE: Overwritten with custom embeddings
            outputs = self.feed_model(model, inputs)
            if has_labels:
                loss, logits = outputs[:2]
                loss = loss.mean().detach()
            else:
                loss = None
                logits = outputs[0]
            if self.args.past_index >= 0:
                self._past = outputs[
                    self.args.past_index if has_labels else self.args.past_index - 1
                ]

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach()
        logits = logits.detach()

        # NOTE: Overwritten for moinitoring purposes (will print occassionally)
        if self.verbose_evaluation and random() < 0.00001:

            try:
                # TODO: Only fill the masked tokens
                prediction = (
                    self.search(logits[1, :, :].unsqueeze(0))
                    .detach()
                    .cpu()
                    .squeeze()
                    .tolist()
                )
                gt_seq, gt_dict = self.tokenizer.aggregate_tokens(
                    self.tokenizer.get_sample_label(
                        self.tokenizer.convert_ids_to_tokens(labels[0]),
                        self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
                    ),
                    label_mode=True,
                )

                p_seq, p_dict = self.tokenizer.aggregate_tokens(
                    self.tokenizer.convert_ids_to_tokens(prediction), label_mode=False
                )

                logger.info(f"\nPredicted: {p_seq} \t, {p_dict.get('qed', -1)}")
                logger.info(f"Ground truth {gt_seq} \t {gt_dict.get('qed', -1)}")
            except Exception:
                logger.info("Error occurred in converting logits to sequence.")

        return loss, logits, labels

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        NOTE: Overwritten to
        1) maintain custom embeddings.
        2) maintain alternating optimization modes

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # NOTE: Overwritten to maintain custom embeddings and alternative losses.
        outputs = self.feed_model(model, inputs)
        loss = outputs[0]

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        return_inputs: bool = False,
        pop_and_return_keys: Optional[List[str]] = None,
        pad_idx: int = 100,
    ) -> PredictionOutput:
        """
        NOTE: Overwritten because
            - fixing tensor stacking https://github.com/huggingface/transformers/issues/7584
            - enable eval_accumulation_steps (introduced only in 3.4.0)
            - to return the inputs

        pop_and_return_keys (Optional[List[str]]): List of keys for the `inputs` dict
            produced by the collator. If passed, each item of list is popped from dict
            *before* calling the model and returned. Defaults to None.

        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(
                dataloader, description, prediction_loss_only=prediction_loss_only
            )
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        if self.args.dataloader_drop_last:
            num_examples -= (num_examples) % batch_size
        num_primed = (
            dataloader.collate_fn.num_primed
            if hasattr(dataloader.collate_fn, "num_primed")
            else 1
        )

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
        inputs_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = 1
        if self.args.local_rank != -1:
            world_size = torch.distributed.get_world_size()
        world_size = max(1, world_size)

        num_examples = num_examples * num_primed
        eval_losses_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
        labels_gatherer = DistributedTensorGatherer(world_size, num_examples)
        inputs_gatherer = DistributedTensorGatherer(world_size, num_examples)

        if pop_and_return_keys:
            return_collator_data = {k: list() for k in NON_MODEL_KEYS}

        # eval_losses: List[float] = []
        # preds: torch.Tensor = None
        # label_ids: torch.Tensor = None
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        # Set up alternating iterator (if applicable)
        alt_loader = self.alt_eval_loader if self.alt_training else dataloader

        # NOTE: The below is a mixture of transformers 3.1.0 and 3.4.0. 3.4.0 introduced
        # the CallbackHandler class which effectively requires rewriting large parts
        # of the package.
        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        samples_count = 0

        epoch_pbar = tqdm(dataloader, desc=description, disable=disable_tqdm)
        for step, (inputs, a_inputs) in enumerate(zip(dataloader, alt_loader)):

            epoch_pbar.update(1)
            # To optionally take out keys from the collator dict.
            if pop_and_return_keys:
                for k in pop_and_return_keys:
                    return_collator_data[k].extend(
                        inputs.get(k, torch.Tensor()).tolist()
                    )
                    inputs.pop(k, None)

            if self.alt_training:
                self.cg_mode = self.get_cg_mode(step)
                if (
                    self.get_cg_mode(step) > self.get_cg_mode(step - 1)
                    and step % 100 == 0
                ):
                    logger.debug("Switching to CG task")
                if self.cg_mode:
                    inputs = a_inputs

            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only
            )

            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            samples_count += batch_size
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = (
                    losses
                    if losses_host is None
                    else torch.cat((losses_host, losses), dim=0)
                )
                # eval_losses.append(loss * batch_size)
            if logits is not None:
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=pad_idx)
                )
            if labels is not None:
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=pad_idx)
                )
            if inputs is not None:
                inputs_host = (
                    inputs["input_ids"]
                    if inputs_host is None
                    else nested_concat(
                        inputs_host, inputs["input_ids"], padding_index=pad_idx
                    )
                )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                self.args.eval_accumulation_steps is not None
                and (step + 1) % self.args.eval_accumulation_steps == 0
            ):
                eval_losses_gatherer.add_arrays(
                    self._gather_and_numpify(losses_host, "eval_losses")
                )
                preds_gatherer.add_arrays(
                    self._gather_and_numpify(preds_host, "eval_preds")
                )
                labels_gatherer.add_arrays(
                    self._gather_and_numpify(labels_host, "eval_label_ids")
                )
                inputs_gatherer.add_arrays(
                    self._gather_and_numpify(inputs_host, "eval_input_ids")
                )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, inputs_host = (
                    None,
                    None,
                    None,
                    None,
                )

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(
            self._gather_and_numpify(losses_host, "eval_losses")
        )
        preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
        labels_gatherer.add_arrays(
            self._gather_and_numpify(labels_host, "eval_label_ids")
        )
        inputs_gatherer.add_arrays(
            self._gather_and_numpify(inputs_host, "eval_input_ids")
        )

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize()
        label_ids = labels_gatherer.finalize()
        input_ids = inputs_gatherer.finalize()

        if (
            self.compute_metrics is not None
            and preds is not None
            and label_ids is not None
        ):
            metrics = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids)
            )
        else:
            metrics = {}
        if eval_loss is not None:
            metrics["eval_loss"] = eval_loss.mean().item()

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        if not return_inputs:
            return PredictionOutput(
                predictions=preds, label_ids=label_ids, metrics=metrics
            )
        elif return_inputs and not pop_and_return_keys:
            return preds, label_ids, metrics, input_ids
        elif return_inputs and pop_and_return_keys:
            return preds, label_ids, metrics, input_ids, return_collator_data

    def _gather_and_numpify(self, tensors, name):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        elif self.args.local_rank != -1:
            tensors = distributed_concat(tensors)

        return nested_numpify(tensors)

    def log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        """
        NOTE: Overwritten to save best model alongside some metrics

        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
            iterator (:obj:`tqdm`, `optional`):
                A potential tqdm progress bar to write the logs on.
        """
        super().log(logs, iterator)

        if "eval_loss" in logs.keys():
            logger.info(f"Evaluation {logs}")
            self.eval_logs.append({"eval_loss": logs["eval_loss"]})
            if "epoch" in logs.keys():
                self.eval_logs[-1].update(
                    {"epoch": logs["epoch"], "step": self.global_step}
                )

        # Custom logging
        if "loss" in logs.keys():
            # In case of training logging
            if self.epoch is not None:
                logs["epoch"] = self.epoch
                output = {**logs, **{"step": self.global_step}}
                self.logs.append(output)

            # Save new best model
            if logs["loss"] < self.min_loss:
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-best-{self.global_step}"
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
                self.min_loss = logs["loss"]
                self.save_model(output_dir)
                # Save optimizer and scheduler
                if self.is_world_master():
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                    torch.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, "scheduler.pt"),
                    )
                pd.DataFrame(self.logs).to_csv(
                    os.path.join(output_dir, "training_log.csv")
                )
                pd.DataFrame(self.eval_logs).to_csv(
                    os.path.join(output_dir, "eval_log.csv")
                )

                checkpoint_prefix = f"{PREFIX_CHECKPOINT_DIR}-best"

                if self.is_world_process_zero():
                    self._rotate_checkpoints(prefix=checkpoint_prefix)

    def _rotate_checkpoints(
        self, use_mtime: bool = False, prefix: str = PREFIX_CHECKPOINT_DIR
    ) -> None:
        """NOTE: Overwritten to enable passing down checkpoint prefix for deletion."""
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=use_mtime, checkpoint_prefix=prefix
        )

        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(
            0, len(checkpoints_sorted) - self.args.save_total_limit
        )
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(
                "Deleting older checkpoint [{}] due to args.save_total_limit".format(
                    checkpoint
                )
            )
            shutil.rmtree(checkpoint)

    def train(
        self,
        model_path: Optional[str] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
    ):
        """
        NOTE: Overwritten to fix a bug in step skipping.

        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            model = self.model_init()
            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps
                // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = int(
                len(train_dataloader)
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(
                    os.path.join(model_path, "optimizer.pt"),
                    map_location=self.args.device,
                )
            )
            self.lr_scheduler.load_state_dict(
                torch.load(os.path.join(model_path, "scheduler.pt"))
            )

        model = self.model

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )

        logger.info("***** Running training *****")
        logger.info(f"Model device {model.device}")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info(
            "  Instantaneous batch size per device = %d",
            self.args.per_device_train_batch_size,
        )
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            total_train_batch_size,
        )
        logger.info(
            "  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split(os.path.sep)[0])
                epochs_trained = self.global_step // (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info(
                    "  Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info(
                    "  Continuing training from global step %d", self.global_step
                )
                logger.info(
                    "  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(
            epochs_trained,
            int(np.ceil(num_train_epochs)),
            desc="Epoch",
            disable=disable_tqdm,
        )
        # NOTE: Fixing a bug where to few steps are skipped.
        steps_to_skip = (
            steps_trained_in_current_epoch * self.args.gradient_accumulation_steps
        )

        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):
            if isinstance(train_dataloader, DataLoader) and isinstance(
                train_dataloader.sampler, DistributedSampler
            ):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Set up alternating iterator (if applicable)
            alt_iterator = (
                self.alt_train_loader if self.alt_training else epoch_iterator
            )

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration", disable=disable_tqdm)
            t = time()
            steps_to_skip = 0
            for step, (inputs, a_inputs) in enumerate(
                zip(epoch_iterator, alt_iterator)
            ):

                # Skip past any already trained steps if resuming training
                if steps_to_skip > 0:
                    steps_to_skip -= 1
                    epoch_pbar.update(1)
                    continue
                logger.debug(f"Step {step}")
                if self.alt_training:
                    self.cg_mode = self.get_cg_mode(step)

                    if self.get_cg_mode(step) > self.get_cg_mode(step - 1):
                        logger.debug(f"Switching to CG task, took {time()-t:.2f}")
                        t = time()
                    elif self.get_cg_mode(step) < self.get_cg_mode(step - 1):
                        logger.debug(f"Switching to PP task, took {time()-t:.2f}")
                        t = time()

                    if self.cg_mode:
                        inputs = a_inputs

                tr_loss += self.training_step(model, inputs).item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.max_grad_norm
                    )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    model.zero_grad()
                    torch.cuda.empty_cache()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (
                        self.args.logging_steps > 0
                        and self.global_step % self.args.logging_steps == 0
                    ) or (self.global_step == 1 and self.args.logging_first_step):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (
                            tr_loss_scalar - logging_loss_scalar
                        ) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)

                    if (
                        self.args.evaluate_during_training
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        metrics = self.evaluate()
                        self.property_evaluate()

                        # The only task that used an XLNetTokenizer was the joke dataset.
                        if isinstance(self.tokenizer, XLNetTokenizer):
                            self.joke_generation_evaluate()

                        self._report_to_hp_search(trial, epoch, metrics)

                    if (
                        self.args.save_steps > 0
                        and self.global_step % self.args.save_steps == 0
                    ):
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert (
                                model.module is self.model
                            ), f"Module {model.module} should be a reference to self.model"
                        else:
                            assert (
                                model is self.model
                            ), f"Model {model} should be a reference to self.model"
                        # Save model checkpoint
                        checkpoint_folder = (
                            f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                        )
                        output_dir = os.path.join(
                            self.args.output_dir, checkpoint_folder
                        )

                        self.save_model(output_dir)

                        if self.is_world_process_zero():
                            self._rotate_checkpoints()

                        if self.is_world_process_zero():
                            torch.save(
                                self.optimizer.state_dict(),
                                os.path.join(output_dir, "optimizer.pt"),
                            )
                            torch.save(
                                self.lr_scheduler.state_dict(),
                                os.path.join(output_dir, "scheduler.pt"),
                            )
                gc.collect()

                epoch_pbar.update(1)
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
            epoch_pbar.close()
            train_pbar.update(1)
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

        train_pbar.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step)

    def get_cg_mode(self, x: int) -> bool:
        """
        For alternating training, we select the mode (PP vs. CG) by alternating every
        `altenate_steps` steps.

        Args:
            x (int): Current step.

        Returns:
            bool: Returns whether or not we are in alternating training mode (i.e.,
                CG optimization) or vanilla training (PP optimization).
        """
        return (x // self.alternate_steps) % 2 == 1

    def property_evaluate(self):
        from terminator.evaluator import Evaluator

        try:
            properties = self.data_collator.property_tokens
        except AttributeError:
            warnings.warn("Collator was not passed explicit properties")
            return

        evaluator = Evaluator(
            model=self.model,
            args=self.args,
            eval_params={},
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            prediction_loss_only=False,
        )
        for pidx, prop in enumerate(properties):
            property_collator = TRAIN_COLLATORS["property"](
                tokenizer=self.tokenizer,
                property_tokens=[prop],
                num_tokens_to_mask=[-1],
            )
            ps, rs, ss = evaluator.property_prediction(
                property_collator, save_path=self.args.output_dir
            )

            if rs[0] < np.min(self.perfs[pidx, 1, :]):
                # Save model
                checkpoint_folder = (
                    f"{PREFIX_CHECKPOINT_DIR}-rmse-min-{self.global_step}"
                )
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
                self.save_model(output_dir)
                # Save optimizer and scheduler
                if self.is_world_master():
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                checkpoint_prefix = f"{PREFIX_CHECKPOINT_DIR}-rmse-min"
                if self.is_world_process_zero():
                    self._rotate_checkpoints(prefix=checkpoint_prefix)

                save_path = os.path.join(output_dir, f"best_rmse_{prop[1:-1]}.json")
                with open(save_path, "w") as f:
                    json.dump({"rmse": rs[0], "pearson": ps[0]}, f)
                logger.info(f"New best RMSE: {rs[0]}")

            if ps[0] > np.max(self.perfs[pidx, 0, :]):
                # Save model
                checkpoint_folder = (
                    f"{PREFIX_CHECKPOINT_DIR}-pearson-max-{self.global_step}"
                )
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
                self.save_model(output_dir)
                # Save optimizer and scheduler
                if self.is_world_master():
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                checkpoint_prefix = f"{PREFIX_CHECKPOINT_DIR}-pearson-max"
                if self.is_world_process_zero():
                    self._rotate_checkpoints(prefix=checkpoint_prefix)

                save_path = os.path.join(output_dir, f"best_pearson_{prop[1:-1]}_.json")
                with open(save_path, "w") as f:
                    json.dump({"rmse": rs[0], "pearson": ps[0]}, f)
                logger.info(f"New best pearson: {ps[0]}")

            if ss[0] > np.max(self.perfs[pidx, 2, :]):
                # Save model
                checkpoint_folder = (
                    f"{PREFIX_CHECKPOINT_DIR}-spearman-max-{self.global_step}"
                )
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
                self.save_model(output_dir)
                # Save optimizer and scheduler
                if self.is_world_master():
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                checkpoint_prefix = f"{PREFIX_CHECKPOINT_DIR}-spearman-max"
                if self.is_world_process_zero():
                    self._rotate_checkpoints(prefix=checkpoint_prefix)

                save_path = os.path.join(
                    output_dir, f"best_spearman_{prop[1:-1]}_.json"
                )
                with open(save_path, "w") as f:
                    json.dump({"rmse": rs[0], "pearson": ps[0], "spearman": ss[0]}, f)
                logger.info(f"New best spearman: {ss[0]}")

            self.perfs[pidx, 0, self.cidx] = ps[0]
            self.perfs[pidx, 1, self.cidx] = rs[0]
            self.perfs[pidx, 2, self.cidx] = ss[0]

        logger.info(f"Current prediction performances {self.perfs[:,:,:self.cidx]}")
        self.cidx += 1

    def joke_generation_evaluate(self, k: int = 10):
        from terminator.evaluator import Evaluator

        logger.info("Entering joke generation evaluation")

        evaluator = Evaluator(
            model=self.model,
            args=self.args,
            eval_params={},
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            prediction_loss_only=False,
        )

        accs, sentences = evaluator.cg_evaluate(dataloader=self.alt_eval_loader, k=k)

        if accs[0] > np.max(self.cg_perfs[0, :]):

            # Save model
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-joke-top1-{self.global_step}"
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            self.save_model(output_dir)
            # Save optimizer and scheduler
            if self.is_world_master():
                torch.save(
                    self.optimizer.state_dict(),
                    os.path.join(output_dir, "optimizer.pt"),
                )
            checkpoint_prefix = f"{PREFIX_CHECKPOINT_DIR}-joke-top1"
            if self.is_world_process_zero():
                self._rotate_checkpoints(prefix=checkpoint_prefix)

            save_path = os.path.join(output_dir, "joke_top1.json")
            with open(save_path, "w") as f:
                json.dump(dict(zip([f"top_{i}" for i in range(k)], accs)), f)
            logger.info(f"New best Top1 joke accuracy: {accs[0]}")

            # Write sentences to file:
            with open(os.path.join(output_dir, "real_jokes.txt"), "w") as f:
                for s in sentences["real"]:
                    f.write(s + "\n")
            for _k in range(k):
                topk_sentences = sentences[f"top_{_k}"]
                with open(os.path.join(output_dir, f"joke_top{_k}.txt"), "w") as f:
                    for s in topk_sentences:
                        f.write(s + "\n")

        self.cg_perfs[0, self.cidx] = accs[0]
        self.cg_perfs[1, self.cidx] = accs[-1]

        logger.info(
            f"Current generation performances top1 and top{k}:{self.cg_perfs[:,:self.cidx]}"
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Overwriting child class method to allow evaluation on the alternating dataloader (CG task)
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.eval_dataset` is a :obj:`torch.utils.data.IterableDataset`, a sequential
        sampler (adapted to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_dataset is not None:
            self._remove_unused_columns(eval_dataset, description="evaluation")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        if self.alternating_collator is not None:
            logger.warning(f"Loading alternative collator for evaluation.")
            return self.alt_eval_loader
        else:
            return DataLoader(
                eval_dataset,
                sampler=eval_sampler,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
            )
