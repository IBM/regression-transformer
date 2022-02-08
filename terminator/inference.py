from typing import Any, Dict, Union

import torch
from transformers import XLNetLMHeadModel

from terminator.factories import NUM_ENCODING_FACTORY
from terminator.tokenization import InferenceBertTokenizer


class InferenceRT:
    def __init__(
        self,
        model: XLNetLMHeadModel,
        tokenizer: InferenceBertTokenizer,
        config: Dict[str, Any] = {},
    ):

        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

        self.params = config.to_dict()
        self.use_ne = self.params['use_ne']

        if self.use_ne:
            self.ne_type = self.params["ne_type"]
            self.ne_format = self.params["ne_format"]

            if self.ne_format == "concat":
                self.combine_embed = self.overwrite_embed
                self.ne_dim = self.params["ne_dim"]
            elif self.ne_format == "sum":
                self.combine_embed = self.sum_embed
                # NE dim has to be identical to real embedding dim
                self.ne_dim = self.params["d_model"]
            else:
                raise ValueError(f"Unknown float encoding format {self.ne_format}.")

            self.numerical_encoder = NUM_ENCODING_FACTORY[self.ne_type](
                num_embeddings=self.params["vocab_size"],
                embedding_dim=self.ne_dim,
                vocab=self.tokenizer.vocab,
                vmax=self.params["vmax"],
            )

            self.model_embed = self.model.transformer.word_embedding

    def sum_embed(self, embed: torch.Tensor, num_embed: torch.Tensor) -> torch.Tensor:
        """
        Summing numerical encodings with regular embeddings

        Args:
            embed: Embedding matrix
            num_embed : Equally sized matrix of numerical encodings. 0 everywhere
                apart from numerical tokens

        Returns:
            torch.Tensor: Summed embedding matrix
        """
        return embed + num_embed

    def overwrite_embed(
        self, embed: torch.Tensor, num_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embed: Embedding matrix
            num_embed: Equally sized matrix of numerical encodings. 0 everywhere
                apart from numerical tokens

        Returns:
            torch.Tensor: Embeding matrix with regular embeddings of last ne_dim
                dimensions replaced by numerical encodings
        """
        embed[:, :, -self.ne_dim :] = num_embed
        return embed

    def __call__(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Forward pass of `inputs` through `model`. This function handles the numerical
        encodings if applicable.

        Args:
            inputs (Dict[str, Union[torch.Tensor, Any]]): A dict that can be understood
                by model.__call__. Keys should include `input_ids`, `perm_mask` etc.
        Returns:
            Dict[str, Union[torch.Tensor, Any]]: Output from model
        """

        if self.use_ne:
            # Pop keys unused by model
            embeddings = self.model_embed(inputs["input_ids"])
            numerical_embeddings = self.numerical_encoder(inputs["input_ids"])
            embeddings = self.combine_embed(embeddings, numerical_embeddings)
            inputs.pop("input_ids", None)
            outputs = self.model(inputs_embeds=embeddings, **inputs)
        else:
            outputs = self.model(**inputs)

        return outputs
