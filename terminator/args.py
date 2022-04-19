from dataclasses import dataclass, field
from typing import Optional

from transformers import MODEL_WITH_LM_HEAD_MAPPING
from transformers.training_args import TrainingArguments

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    NOTE: Expanding TrainingArguments class from transformers with custom arguments.

    eval_accumulation_steps (:obj:`int`, `optional`):
            Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
            left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but
            requires more memory).
    """

    # Was introduced only in transformers 3.4.0
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of predictions steps to accumulate before moving the tensors to the CPU."
        },
    )
    training_config_path: Optional[str] = field(
        default=None,
        metadata={
            "help": """
            Path to a file specifying the training objective hyperparameter.

            Defaults to None, meaning the vanilla PLM objective is used.


            Optional keys include:
            - 'alternate_tasks' (bool): Whether the model is trained specifically on
                property prediction and conditional generation task or not.
                NOTE: If False, then all other keys are ignored and we fall back to the
                PLM objective (identical to not providing a path). Default: False.
            - 'cc_loss' (bool): Whether the model is trained with the cycle-consistency
                loss in the CG task or with a regular BCE between logits of generated
                tokens and the real molecule. Default: False.
            - 'cg_collator' (str): Name of collator to use for conditional generation.
                Should be either `vanilla_cg` or `bimodal_cg`.
            - 'generation_token' (str): Token which should be masked for CC loss. Only
                required if cc_loss is True.

            - 'cg_collator_params' (dict): Parameters to pass to the collator. Keys e.g.
                'do_sample' (bool): Whether property is sampled.
                'property_value_ranges' (Iterable[float]):
                'property_value_thresholds' (Iterable[float]):
                'prob_near_sampling' (float): Probability of sampling nearby values.
            """
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class EvalArguments:
    """
    Argumnts for model evaluation.

    eval_accumulation_steps (:obj:`int`, `optional`):
            Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
            left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but
            requires more memory).
    """

    eval_file: str = field(metadata={"help": "Path to the data used for evaluation"})
    param_path: str = field(
        metadata={"help": "Path to the .json file with evaluation parameter"}
    )
