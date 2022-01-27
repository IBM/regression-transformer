from .numerical_encodings import FloatEncoding, IntEncoding

NUM_ENCODING_FACTORY = {"float": FloatEncoding, "int": IntEncoding}

MODEL_TO_EMBEDDING_FN = {
    "albert": "model.albert.embeddings",
    "xlnet": "self.model.transformer.word_embedding",
}
