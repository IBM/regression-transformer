"""
Taken from https://github.com/huggingface/transformers/blob/v3.1.0/src/transformers/trainer.py
"""
import os
from time import time
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from selfies import encoder
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForPermutationLanguageModeling
from transformers.utils import logging

from terminator.collators import PropertyCollator
from terminator.nlp import compute_topk
from terminator.search import BeamSearch, GreedySearch, SamplingSearch
from terminator.trainer import CustomTrainer
from terminator.utils import find_safe_path

logger = logging.get_logger(__name__)


class Evaluator(CustomTrainer):
    def __init__(
        self,
        eval_params,
        *args,
        **kwargs,
    ):

        # Call parent class constructor
        super().__init__(*args, **kwargs)

        self.eval_params = eval_params

        self.greedy_search = GreedySearch()
        self.sampling_search = SamplingSearch(
            temperature=eval_params.get("temperature", 1.0)
        )
        self.beam_search = BeamSearch(
            temperature=eval_params.get("temperature", 1.0),
            beam_width=eval_params.get("beam_width", 1),
            top_tokens=eval_params.get("beam_top_tokens", 5),
        )

    def get_custom_dataloader(self, collator, bs: int = -1) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.eval_dataset` is a :obj:`torch.utils.data.IterableDataset`, a sequential
        sampler (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`nlp.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed.
        """
        if not isinstance(collator, DataCollatorForPermutationLanguageModeling):
            raise TypeError(f"Needs PLM collator not {type(collator)}")
        return DataLoader(
            self.eval_dataset,
            sampler=self._get_eval_sampler(self.eval_dataset),
            batch_size=self.args.eval_batch_size if bs == -1 else bs,
            collate_fn=collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def multi_property_prediction(self, collator, save_path=None, rmse_factor: int = 1):

        eval_dataloader = self.get_custom_dataloader(collator)
        # Forward pass
        logits, label_ids, metrics, input_ids = self.prediction_loop(
            dataloader=eval_dataloader,
            description=f"Predicting {collator.property_tokens}",
            prediction_loss_only=False,
            return_inputs=True,
            pad_idx=self.tokenizer.vocab["[PAD]"],
        )
        keep_pos = lambda x: not np.logical_or((x == -100), (x == 0)).all()
        pos_to_keep = [i for i in range(logits.shape[1]) if keep_pos(label_ids[:, i])]
        relevant_logits = torch.Tensor(logits[:, pos_to_keep, :])

        # Compute performance (only on the logits where predictions are relevavnt)
        print("Obtained logits, now performing beam search...")
        greedy_preds = self.greedy_search(relevant_logits).unsqueeze(0)
        sampling_preds = self.sampling_search(relevant_logits).unsqueeze(0)
        beam_preds, scores = self.beam_search(relevant_logits)
        beam_preds = beam_preds.permute(2, 0, 1)

        # Reassign full prediction matrices (0 means mask)
        all_preds = torch.zeros(2 + beam_preds.shape[0], *logits.shape[:2]).long()
        all_preds[0, :, pos_to_keep] = greedy_preds
        all_preds[1, :, pos_to_keep] = sampling_preds
        # In case beam width > 0:
        for k in range(beam_preds.shape[0]):
            all_preds[2 + k, :, pos_to_keep] = beam_preds[k, :, :]

        # Define rmse function
        rmse = lambda x, y: np.sqrt(sum((np.array(x) - np.array(y)) ** 2) / len(x))

        num_props = len(collator.property_tokens)
        num_decoders = 2 + beam_preds.shape[0]
        num_samples = len(relevant_logits)

        property_labels = torch.zeros(num_props, num_samples)
        property_predictions = torch.zeros(num_props, num_decoders, num_samples)
        pearsons = np.zeros((num_props, num_decoders))
        rmses = np.zeros((num_props, num_decoders))
        for idx, predictions in enumerate(all_preds):

            for sidx, (x, y, yhat) in enumerate(zip(input_ids, label_ids, predictions)):

                x_tokens = self.tokenizer.decode(
                    x, clean_up_tokenization_spaces=False
                ).split(" ")
                y_tokens = self.tokenizer.decode(
                    y, clean_up_tokenization_spaces=False
                ).split(" ")
                yhat_tokens = self.tokenizer.decode(
                    yhat, clean_up_tokenization_spaces=False
                ).split(" ")

                # Get label (float)
                label = self.tokenizer.get_sample_label(y_tokens, x_tokens)
                # Get prediction (float)
                gen = self.tokenizer.get_sample_prediction(yhat_tokens, x_tokens)

                _, target_prop = self.tokenizer.aggregate_tokens(label, label_mode=True)
                _, gen_prop = self.tokenizer.aggregate_tokens(gen, label_mode=False)

                assert target_prop.keys() == gen_prop.keys()
                # Save predictions for all properties
                for i, p in enumerate(target_prop.keys()):
                    if idx == 0:
                        property_labels[i, sidx] = target_prop[p] / rmse_factor
                    property_predictions[i, idx, sidx] = gen_prop[p] / rmse_factor

            for i, prop in enumerate(collator.property_tokens):
                p = pearsonr(property_labels[i, :], property_predictions[i, idx, :])
                r = rmse(property_labels[i, :], property_predictions[i, idx, :])
                if idx == 0:
                    print(f"Pearson for {prop} is {p[0]:.3f}")
                    print(f"RMSE for {prop} is {r:.3f}")
                pearsons[i, idx] = p[0]
                rmses[i, idx] = r

        if save_path is not None:

            bw = self.beam_search.beam_width
            beam_cols = ["Beam"] if bw == 1 else [f"Beam{i}" for i in range(bw)]
            search_cols = ["Label", "Greedy", "Sampling"] + beam_cols
            for i, prop in enumerate(collator.property_tokens):
                pd.DataFrame(
                    np.concatenate(
                        [
                            property_labels[i, :].unsqueeze(1).numpy(),
                            property_predictions[i, :, :].T.numpy(),
                        ],
                        axis=1,
                    ),
                    columns=search_cols,
                ).to_csv(f"{save_path}_predict_{prop[1:-1]}.csv")

        return pearsons, rmses

    def property_prediction(self, collator, save_path=None, rmse_factor: int = 1):
        """
        Predict the property
        """

        eval_dataloader = self.get_custom_dataloader(collator)

        # Hack for XLNet tokenizer
        self.tokenizer.real_decoder = self.tokenizer.decode
        self.tokenizer.decode = self.tokenizer.decode_internal

        for prop in collator.property_tokens:

            # Forward pass
            logits, label_ids, metrics, input_ids = self.prediction_loop(
                dataloader=eval_dataloader,
                description=f"Predicting {prop}",
                prediction_loss_only=False,
                return_inputs=True,
                pad_idx=self.tokenizer.vocab["[PAD]"],
            )
            _prop = prop[1:-1]

            """
            NOTE: Saving time by only running search on affected positions
            Sequence positions where the label was not -100 (MASK) or 0 (PAD) at least
            once. Those positions are used for the searches. This heavily bases on the
            assumption that the positions are more or less *stable* across the samples
            (good for property prediction but for CD, it's less efficient).
            """
            keep_pos = lambda x: not np.logical_or((x == -100), (x == 0)).all()
            pos_to_keep = [
                i for i in range(logits.shape[1]) if keep_pos(label_ids[:, i])
            ]
            relevant_logits = torch.Tensor(logits[:, pos_to_keep, :])

            # Compute performance (only on the logits where predictions are relevavnt)
            print("Obtained logits, now performing beam search...")
            greedy_preds = self.greedy_search(relevant_logits).unsqueeze(0)
            sampling_preds = self.sampling_search(relevant_logits).unsqueeze(0)
            # beam_preds, scores = self.beam_search(relevant_logits)
            # beam_preds = beam_preds.permute(2, 0, 1)

            # Reassign full prediction matrices (0 means mask)
            all_preds = torch.zeros(2, *logits.shape[:2]).long()
            all_preds[0, :, pos_to_keep] = greedy_preds
            all_preds[1, :, pos_to_keep] = sampling_preds
            # In case beam width > 0:
            # for k in range(beam_preds.shape[0]):
            #     all_preds[2 + k, :, pos_to_keep] = beam_preds[k, :, :]

            # Define rmse function
            rmse = lambda x, y: np.sqrt(sum((np.array(x) - np.array(y)) ** 2) / len(x))

            property_labels = torch.zeros(len(relevant_logits))
            property_predictions = torch.zeros(len(relevant_logits), len(all_preds))
            pearsons, rmses, spearmans = (
                np.zeros((len(all_preds))),
                np.zeros((len(all_preds))),
                np.zeros((len(all_preds))),
            )
            for idx, predictions in enumerate(all_preds):

                for sidx, (x, y, yhat) in enumerate(
                    zip(input_ids, label_ids, predictions)
                ):

                    x_tokens = self.tokenizer.decode(
                        x, clean_up_tokenization_spaces=False
                    ).split(" ")
                    y_tokens = self.tokenizer.decode(
                        y, clean_up_tokenization_spaces=False
                    ).split(" ")
                    yhat_tokens = self.tokenizer.decode(
                        yhat, clean_up_tokenization_spaces=False
                    ).split(" ")
                    # print('yhat tokens', ''.join(yhat_tokens))
                    # print('y_tokens', ''.join(y_tokens))
                    # print('y', y)

                    # Get label (float)
                    label = self.tokenizer.get_sample_label(y_tokens, x_tokens)
                    # Get prediction (float)
                    gen = self.tokenizer.get_sample_prediction(yhat_tokens, x_tokens)

                    _, target_prop = self.tokenizer.aggregate_tokens(
                        label, label_mode=True
                    )
                    # print('target_prop is', target_prop)
                    _, gen_prop = self.tokenizer.aggregate_tokens(gen, label_mode=False)
                    # print('gen_prop is', gen_prop)

                    property_predictions[sidx, idx] = (
                        gen_prop.get(_prop, -1) / rmse_factor
                    )
                    if idx == 0:
                        property_labels[sidx] = target_prop[_prop] / rmse_factor

                p = pearsonr(property_labels, property_predictions[:, idx])
                r = rmse(property_labels, property_predictions[:, idx])
                s = spearmanr(property_labels, property_predictions[:, idx])
                if idx == 0:
                    print(f"Pearson is {p[0]}")
                    print(f"RMSE is {r}")
                    print(f"Spearman is {s[0]}")
                else:
                    print("SAMPLING Preds")
                    print(f"Pearson is {p[0]}")
                    print(f"RMSE is {r}")
                    print(f"Spearman is {s[0]}")
                pearsons[idx] = p[0]
                rmses[idx] = r
                spearmans[idx] = s[0]
            if save_path is not None:
                bw = self.beam_search.beam_width
                bw = 0
                beam_cols = ["Beam"] if bw == 1 else [f"Beam{i}" for i in range(bw)]
                search_cols = ["Label", "Greedy", "Sampling"] + beam_cols
                pd.DataFrame(
                    np.concatenate(
                        [
                            property_labels.unsqueeze(1).numpy(),
                            property_predictions.numpy(),
                        ],
                        axis=1,
                    ),
                    columns=search_cols,
                ).to_csv(
                    os.path.join(save_path, "checkpoint-rmse-min-2400", "test.csv")
                )

        self.tokenizer.decode = self.tokenizer.real_decoder

        return pearsons, rmses, spearmans

    def conditional_generation(
        self,
        collator,
        save_path: str = None,
        passed_eval_fn: Callable = None,
        property_collator=None,
        denormalize_params: Optional[List[float]] = None,
    ):
        """
        Function to evaluate conditional generation

        Args:
            collator (): PyTorch collator object
            save_path (str): Path where results are saved, defaults to None (no saving).
            passed_eval_fn (Callable): Function used to evaluate whether the generated molecules
                adhere to the property of interest. Defaults to None, in which case the model
                is used *itself* to evaluate the molecules (NOTE: This can be a biased estimator).
                NOTE: Function should be callable with a SMILES string.
            property_collator (): PyTorch collator object. Defaults to None. Only needed if passed_eval_fn
                is None
            denormalize_params: The min and max values of the property to denormalize
                the results.
        """

        if passed_eval_fn is None and property_collator is None:
            raise ValueError(
                "If model should be used for evaluation, property collator is required"
            )

        eval_dataloader = self.get_custom_dataloader(collator, bs=2)
        prop = collator.property_token[1:-1]

        if passed_eval_fn is not None:
            eval_fn = passed_eval_fn
        else:
            eval_fn = self.get_seq_eval_fn(
                collator=property_collator, prefix=f"<{prop}>0.123|"
            )
        if denormalize_params:
            denormalize = (
                lambda x: x * (denormalize_params[1] - denormalize_params[0])
                + denormalize_params[0]
            )
        else:
            denormalize = lambda x: x

        seq_to_prop = {}

        # Forward pass
        logits, label_ids, metrics, input_ids, returned = self.prediction_loop(
            dataloader=eval_dataloader,
            description=f"Conditional generation {prop}",
            prediction_loss_only=False,
            return_inputs=True,
            pop_and_return_keys=["real_property", "sample_weights"],
            pad_idx=self.tokenizer.vocab["[PAD]"],
        )
        logits = torch.Tensor(logits).cpu()
        input_ids = torch.Tensor(input_ids)
        # Arbitrary rounding set here
        real_prop = [round(denormalize(x), 4) for x in returned["real_property"]]

        # Naive search (using all tokens)
        t = time()
        greedy_preds = self.greedy_search(logits).unsqueeze(0)
        logger.error(f"Greedy search took {time() - t}")
        t = time()
        sampling_preds = self.sampling_search(logits).unsqueeze(0)
        logger.error(f"Sampling search took {time() - t}")
        # Just needed for reference
        bw = self.beam_search.beam_width
        beam_preds = torch.cat([greedy_preds] * bw, dim=0)

        # Restrict beam search to affected logits
        t = time()
        for sample_idx in tqdm(range(logits.shape[0]), desc="Beam search"):
            keep_pos = torch.nonzero(
                input_ids[sample_idx, :] == self.tokenizer.vocab["[MASK]"]
            ).squeeze(1)
            relevant_logits = logits[sample_idx, keep_pos, :].unsqueeze(0)
            if len(relevant_logits.shape) == 2:
                relevant_logits = relevant_logits.unsqueeze(0)
            beams, scores = self.beam_search(relevant_logits)
            beam_preds[:, sample_idx, keep_pos] = (
                beams.squeeze(dim=0).permute(1, 0).long()
            )
        print(f"Beam search took {time() - t}")

        all_preds = torch.zeros(2 + beam_preds.shape[0], *logits.shape[:2]).long()
        all_preds[0, :, :] = greedy_preds
        all_preds[1, :, :] = sampling_preds

        # In case beam width > 0:
        if bw > 0:
            for k in range(beam_preds.shape[0]):
                all_preds[2 + k, :, :] = beam_preds[k, :, :]

        array_len = logits.shape[0] * all_preds.shape[0]
        property_primers = torch.zeros(array_len)
        property_generations = torch.zeros(array_len)
        original_seqs = np.empty(array_len, dtype="object")
        generated_seqs = np.empty(array_len, dtype="object")
        prop_dicts = []

        for idx, predictions in tqdm(enumerate(all_preds), desc="Evaluating search"):
            for sidx, (x, y, yhat) in tqdm(
                enumerate(zip(input_ids, label_ids, predictions))
            ):

                cidx = idx * len(predictions) + sidx
                assert len(x) == len(y), "Input and label lengths do not match"
                assert len(x) == len(yhat), "Input and predictions do not match"
                x_tokens = self.tokenizer.decode(
                    x, clean_up_tokenization_spaces=False
                ).split(" ")
                y_tokens = self.tokenizer.decode(
                    y, clean_up_tokenization_spaces=False
                ).split(" ")
                yhat_tokens = self.tokenizer.decode(
                    yhat, clean_up_tokenization_spaces=False
                ).split(" ")
                assert len(x_tokens) == len(
                    y_tokens
                ), f"I/O lengths must match  {len(x_tokens)} and {len(y_tokens)}"
                # Get property primer (float)
                label = self.tokenizer.get_sample_label(y_tokens, x_tokens)
                # Get prediction (float)
                gen = self.tokenizer.get_sample_prediction(yhat_tokens, x_tokens)
                orgseq, target_prop = self.tokenizer.aggregate_tokens(
                    label, label_mode=True
                )
                original_seqs[cidx] = orgseq.split("|")[-1]

                genseq, _ = self.tokenizer.aggregate_tokens(gen, label_mode=False)
                generated_seqs[cidx] = genseq.split("|")[-1]
                gen_seq = self.tokenizer.to_readable(generated_seqs[cidx])

                # Checking whether molecule was already predicted
                if gen_seq in seq_to_prop.keys():
                    prop_dict = seq_to_prop[gen_seq]
                    value = prop_dict[prop.lower()]
                else:
                    value, prop_dict = eval_fn(gen_seq)
                    value = denormalize(value)
                    prop_dict[prop.lower()] = round(value, 4)
                prop_dicts.append(prop_dict)

                # except Exception:
                #     value = -1
                #     prop_dicts.append({})
                seq_to_prop[gen_seq] = prop_dict

                property_generations[cidx] = value
                property_primers[cidx] = denormalize(target_prop[prop])

        pg = property_generations[property_generations != -1]
        print(f"Ratio of invalid sequences: {round(1 - (len(pg)/array_len),2)}")
        pp = property_primers[property_generations != -1]
        p = pearsonr(pp, pg)
        s = spearmanr(pp, pg)

        print(f"Global Pearson is {round(p[0], 3)} ({p[1]})")
        print(f"Global Spearman is {round(s[0], 3)} ({s[1]})")

        if save_path is not None:
            beam_cols = ["Beam"] if bw == 1 else [f"Beam{i}" for i in range(bw)]
            search_cols = ["Greedy", "Sampling"] + beam_cols
            prop = prop.capitalize()
            save_path = find_safe_path(save_path)

            df = pd.DataFrame(
                {
                    "SeedSequence": original_seqs,
                    f"Seed{prop}": np.tile(
                        np.repeat(real_prop, collator.num_primed), len(search_cols)
                    ),
                    f"Primer{prop}": property_primers,
                    "GenSequence": generated_seqs,
                    f"Gen{prop}": property_generations.tolist(),
                    "Search": np.repeat(
                        search_cols, collator.num_primed * len(real_prop)
                    ),
                }
            )
            remaining_props = pd.DataFrame(prop_dicts)
            replacer = dict(
                zip(
                    remaining_props.columns,
                    [f"Gen{k.capitalize()}" for k in remaining_props.columns],
                )
            )
            remaining_props = remaining_props.rename(columns=replacer)
            remaining_props = remaining_props.drop(columns=[f"Gen{prop}"])
            df = pd.concat([df, remaining_props], axis=1)
            df["sort_helper"] = df.Search.apply(
                lambda x: {"Greedy": 0, "Sampling": 2}.get(x, 1)
            )
            df = df.sort_values(
                by=["SeedSequence", "Search", f"Primer{prop}"],
                ascending=[True, False, True],
            ).drop_duplicates(subset=["GenSequence", "SeedSequence"])
            df.to_csv(os.path.join(save_path))
            print(f"Data frame has {len(df)} samples, saved in {save_path}")
            spearmans = [
                spearmanr(
                    df[(df.SeedSequence == x) & (df.Search == "Sampling")][
                        f"Primer{prop}"
                    ],
                    df[(df.SeedSequence == x) & (df.Search == "Sampling")][
                        f"Gen{prop}"
                    ],
                )[0]
                for x in df.SeedSequence.unique()
            ]
            print(
                f"Average per sample Spearman for sampling search: {np.nanmean(spearmans)}"
            )

    def get_seq_eval_fn(self, collator: PropertyCollator, prefix: str) -> Callable:
        """
        Returns a function that can be called with a sequence and returns the predicted
        property by the model. The property token is being set by the collator.

        Args:
            collator (PropertyCollator): A property collator

        Returns:
            Callable: Function to be called with a text string
        """
        from terminator.property_predictors import PREDICT_FACTORY

        property_token = collator.property_tokens[0]
        # Check whether shortcut is possible
        if property_token in PREDICT_FACTORY.keys():
            return PREDICT_FACTORY[property_token]

        def eval_fn(seq):
            if not isinstance(seq, str):
                return -1, {property_token: -1}
            if self.tokenizer.language == "SELFIES":
                seq = encoder(seq)
            input_str = f"{prefix}{seq}"
            sample_ids = [torch.tensor(self.tokenizer(input_str)["input_ids"])]
            prepared_inputs = collator(sample_ids)
            _, logits, _ = self.prediction_step(
                model=self.model, inputs=prepared_inputs, prediction_loss_only=False
            )
            greedy_preds = self.greedy_search(logits).squeeze(0)
            x_tokens = self.tokenizer.decode(
                prepared_inputs["input_ids"].squeeze()
            ).split(" ")
            yhat_tokens = self.tokenizer.decode(greedy_preds).split(" ")
            gen = self.tokenizer.get_sample_prediction(yhat_tokens, x_tokens)
            _, prop_dict = self.tokenizer.aggregate_tokens(gen, label_mode=False)
            value = prop_dict[property_token[1:-1]]
            return value, prop_dict

        return eval_fn

    def cg_evaluate(self, dataloader, k: int = 10):
        """
        Function to evaluate funnyness dataset joke generation
        """

        # Forward pass
        logits, label_ids, metrics, input_ids = self.prediction_loop(
            dataloader=dataloader,
            description="Conditional generation",
            prediction_loss_only=False,
            return_inputs=True,
            pad_idx=self.tokenizer.vocab["[PAD]"],
        )
        num_samples = logits.shape[0]
        logits = torch.Tensor(logits).cpu()
        input_ids = torch.Tensor(input_ids)

        topk_values, topk_indices = torch.topk(logits, k=k, dim=2)
        all_preds = topk_indices.permute(2, 0, 1)

        # self.tokenizer.decode = self.tokenizer.decode_internal
        accuracies = np.zeros((num_samples, k))
        sentences = {"real": []}
        for _k in range(k):
            sentences[f"top_{_k}"] = []

        for idx, predictions in tqdm(enumerate(all_preds), desc="Evaluating search"):
            for sidx, (x, y, yhat) in tqdm(
                enumerate(zip(input_ids, label_ids, predictions))
            ):
                cidx = idx * len(predictions) + sidx
                x[x == -100] = 6
                y[y == -100] = 6
                yhat[yhat == -100] = 6
                assert len(x) == len(y), "Input and label lengths do not match"
                assert len(x) == len(yhat), "Input and predictions do not match"
                x_tokens = self.tokenizer.decode(
                    x, clean_up_tokenization_spaces=False
                ).split(" ")

                y_tokens = self.tokenizer.decode(
                    y, clean_up_tokenization_spaces=False
                ).split(" ")
                _x_tokens = []
                for x in x_tokens:
                    if "{" in x and "}" in x and "<mask>" in x:
                        _x_tokens.extend(["{"] + ["<mask>"] * x.count("<mask>") + ["}"])
                    elif "<sep>" in x:
                        _x_tokens.append(x.split("<sep>")[0])
                    else:
                        _x_tokens.append(x)

                x_tokens = []
                for x in _x_tokens:
                    if "[PAD]" in x:
                        break
                    x_tokens.append(x)
                _y_tokens = []
                for y in y_tokens:
                    if y.startswith("<mask>"):
                        _y_tokens.extend(y.split("<mask>")[1:])
                    else:
                        _y_tokens.extend(y.split("<mask>"))
                y_tokens = _y_tokens

                # Get sample prediction
                s, e = x_tokens.index(
                    self.tokenizer.expression_separator
                ), x_tokens.index(self.tokenizer.expression_end)
                predicted = self.tokenizer.decode(
                    yhat[s + 1 : e], clean_up_tokenization_spaces=False
                )
                predicted = predicted.split(" ") if predicted != "" else [""]
                predicted = "".join(predicted)

                label = "".join(y_tokens[s + 1 : e])
                if predicted == label:
                    accuracies[sidx, idx] = 1

                # Real sentence
                real_sentence = x_tokens[e + 1 :]
                joke_idxs = list(
                    range(real_sentence.index("START") + 1, real_sentence.index("END"))
                )
                predicted_sentence = []
                filled = False
                for i, x in enumerate(real_sentence):
                    if (i not in joke_idxs) or filled:
                        predicted_sentence.append(x)
                    else:
                        # Fill joke
                        predicted_sentence.append(predicted)
                        filled = True

                # Change real sentence s.t. the joke is replaced with ground truth
                for i, j in enumerate(joke_idxs):
                    real_sentence[j] = y_tokens[s + 1 + i]

                real_sentence = " ".join(
                    [x for x in real_sentence if x not in ["START", "END"]]
                )

                predicted_sentence = " ".join(
                    [x for x in predicted_sentence if x not in ["START", "END"]]
                )
                if idx == 0:
                    sentences["real"].append(real_sentence)
                sentences[f"top_{idx}"].append(predicted_sentence)

        topk_accuracies = compute_topk(accuracies)
        # self.tokenizer.decode = self.tokenizer.real_decode
        return topk_accuracies, sentences
