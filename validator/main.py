import itertools
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np

from guardrails.utils.docs_utils import get_chunks_from_text
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
    OnFailAction
)
from llama_index.embeddings.openai import OpenAIEmbedding

from constants import DEFAULT_FEW_SHOT_TRAIN_PROMPTS, ARIZE_DEFAULT_RESPONSE


def _embed_function(text: Union[str, List[str]]) -> np.ndarray:
    """Function used to embed text with OpenAIEmbedding(model="text-embedding-ada-002").

    :param text: Either a string or list of strings that will be embedded.

    :return: Array of embedded input string(s).
    """
    if isinstance(text, str):
        text = [text]

    embeddings_out = []
    for current_example in text:
        embedding = OpenAIEmbedding(model="text-embedding-ada-002").get_text_embedding(current_example)
        embeddings_out.append(embedding)
    return np.array(embeddings_out)


class EmbeddingChunkStrategy(Enum):
    """Chunk strategy used in get_chunks_from_text when creating embeddings."""
    SENTENCE = 0
    WORD = 1
    CHAR = 2
    TOKEN = 3


@register_validator(name="arize/dataset_embeddings", data_type="string")
class ArizeDatasetEmbeddings(Validator):
    """Validates that user-generated input does not match dataset of jailbreak
    embeddings from Arize AI."""

    def __init__(
        self,
        threshold: float = 0.2,
        default_response: str = ARIZE_DEFAULT_RESPONSE,
        on_fail: Optional[Union[Callable, OnFailAction]] = OnFailAction.FIX,
        sources: List[str] = DEFAULT_FEW_SHOT_TRAIN_PROMPTS,
        embed_function: Callable = _embed_function,
        chunk_strategy: EmbeddingChunkStrategy = EmbeddingChunkStrategy.SENTENCE,
        chunk_size: int = 100,
        chunk_overlap: int = 20,
        **kwargs,
    ):
        super().__init__(
            on_fail=on_fail, threshold=threshold, **kwargs
        )
        self._default_response = default_response
        self._threshold = float(threshold)
        self._sources = sources
        self._embed_function = embed_function
        
        # Validate we have a non-empty dataset containing string messages
        for prompt in self._sources:
            if not prompt or not isinstance(prompt, str):
                raise ValueError(f"Prompt example: {prompt} is invalid. Must contain valid string data.")
        
        chunks = [
            get_chunks_from_text(source, chunk_strategy.name.lower(), chunk_size, chunk_overlap)
            for source in self._sources
        ]
        self._chunks = list(itertools.chain.from_iterable(chunks))

        # Create embeddings
        self._source_embeddings = np.array(self._embed_function(self._chunks)).squeeze()

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        """Validation function for the ArizeDatasetEmbeddings validator. If the cosine distance
        of the user input embeddings is below the user-specified threshold of the closest embedded chunk
        from the jailbreak examples in prompt sources, then the Guard will return FailResult. If all chunks
        are sufficiently distant, then the Guard will return PassResult.

        :param value: This is the 'value' of user input. For the ArizeDatasetEmbeddings Guard, we want
            to ensure we are validating user input, rather than LLM output, so we need to call
            the guard with Guard().use(ArizeDatasetEmbeddings, on="prompt")

        :return: PassResult or FailResult.
        """
        # Get user message if available explicitly as metadata. If unavailable, use value. This could be
        # the context, prompt or LLM output, depending on how the Guard is set up and called.
        user_message = metadata.get("user_message", value)
        
        # Get closest chunk in the embedded few shot examples of jailbreak prompts.
        # Get cosine distance between the embedding of the user message and the closest embedded jailbreak prompts chunk.
        closest_chunk, lowest_distance = self.query_vector_collection(text=user_message, k=1)[0]
        metadata["lowest_cosine_distance"] = lowest_distance
        metadata["most_similar_dataset_chunk"] = closest_chunk
        
        # Pass or fail Guard based on minimum cosine distance between user message and embedded jailbreak prompts.
        if lowest_distance < self._threshold:
            # At least one jailbreak embedding chunk was within the cosine distance threshold from the user input embedding
            return FailResult(
                metadata=metadata,
                error_message=(
                    f"The following message triggered the ArizeDatasetEmbeddings Guard:\n\t{user_message}"
                ),
                fix_value=self._default_response
            )
        # All chunks exceeded the cosine distance threshold
        return PassResult(metadata=metadata)

    def query_vector_collection(
            self,
            text: str,
            k: int,
    ) -> List[Tuple[str, float]]:
        """Embed user input text and compute cosine distances to prompt source embeddings (jailbreak examples).

        :param text: Text string from user message. This will be embedded, then we will calculate the cosine distance
            to each embedded chunk in our prompt source embeddings. 

        :return: List of tuples containing the closest chunk (string text) and the float distance between that 
            embedded chunk and the user input embedding.
        """

        # Create embeddings on user message
        query_embedding = self._embed_function(text).squeeze()

        # Compute distances
        cos_distances = 1 - (
            np.dot(self._source_embeddings, query_embedding)
            / (
                np.linalg.norm(self._source_embeddings, axis=1)
                * np.linalg.norm(query_embedding)
            )
        )
        
        # Sort indices from lowest cosine distance to highest distance
        low_to_high_ind = np.argsort(cos_distances)[:k]
        
        # Get top-k closest distances
        lowest_distances = [cos_distances[j] for j in low_to_high_ind]

        # Get top-k closest chunks
        closest_chunks = [self._chunks[j] for j in low_to_high_ind]

        return list(zip(closest_chunks, lowest_distances))


