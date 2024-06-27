import itertools
import numpy as np
from functools import partial
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
)
from llama_index.embeddings.openai import OpenAIEmbedding


def _embed_function(text: Union[str, List[str]]) -> np.ndarray:
    """Function used to embed text"""
    if isinstance(text, str):
        text = [text]

    embeddings_out = []
    for current_example in text:
        embeddings_out.append(OpenAIEmbedding(model="text-embedding-ada-002").get_text_embedding(current_example))
    return np.array(embeddings_out)


class EmbeddingChunkStrategy(Enum):
    """Chunk strategy used in get_chunks_from_text when creating embeddings."""
    SENTENCE = 0
    WORD = 1
    CHAR = 2
    TOKEN = 3


@register_validator(name="arize/jailbreak_embeddings", data_type="string")
class JailbreakEmbeddings(Validator):
    """Validates that user-generated input does not match dataset of jailbreak
    embeddings from Arize AI."""

    def __init__(
            self,
            prompt_sources: List[str],
            threshold: float = 0.2,
            on_fail: Optional[Callable] = None,
            embed_function: Optional[Callable] = _embed_function,
            chunk_strategy: EmbeddingChunkStrategy = EmbeddingChunkStrategy.SENTENCE,
            chunk_size: int = 100,
            chunk_overlap: int = 20,
            **kwargs,
    ):
        """Initialize Arize AI Guard against jailbreak embeddings.

        :param prompt_sources: Examples of jailbreak attempts that we want to Guard against. These examples will
            be embedded by our embed_function. If the Guard sees a user input message with embeddings that are close
            to any of the embedded chunks from our jailbreak examples, then the Guard will flag the jailbreak attempt.
            We recommend adding at least 10 examples.
        :param threshold: Float values between 0.0 and 1.0. Defines the threshold at which a new user input is close
            enough to an embedded prompt_sources chunk that the Guard flags a jailbreak attempt. The distance is measured
            as the cosine distance between embeddings.
        :on_fail: Inherited from Validator.
        :embed_function: Embedding function used to embed both the prompt_sources and live user input.
        :chunk_strategy: The strategy to use for chunking when calling Guardrails AI. Strategies include sentence,
            word, character or token. Details in get_chunks_from_text.
        :chunk_size: Usage defined by Guardrails AI. The size of each chunk. If the chunk_strategy is "sentences",
            this is the number of sentences per chunk. If the chunk_strategy is "characters", this is the number of 
            characters per chunk, and so on. Defaults to 100 through trial-and-error.
        :chunk_overlap: The number of characters to overlap between chunks. If the chunk_strategy is "sentences", this 
            is the number of sentences to overlap between chunks. Defaults to 20 through trial-and-error.
        """
        super().__init__(on_fail, **kwargs)
        
        self._threshold = float(threshold)
        self.sources = prompt_sources

        chunks = [
            get_chunks_from_text(source, chunk_strategy.name.lower(), chunk_size, chunk_overlap)
            for source in self.sources
        ]
        self.chunks = list(itertools.chain.from_iterable(chunks))

        # Create embeddings
        self.source_embeddings = np.array(self.embed_function(self.chunks)).squeeze()

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        """Validation function for the ProvenanceEmbeddings validator."""
        # Replace LLM response with user input prompt
        print("THIS IS WHAT THE VALUE IS {}".format(value))
        most_similar_chunks = self.query_vector_collection(text=value, k=1)
        if most_similar_chunks is None:
            metadata["highest_similarity_score"] = 0
            metadata["similar_jailbreak_phrase"] = ""
            return PassResult(metadata=metadata)

        closest_chunk, lowest_distance = most_similar_chunks[0]
        metadata["highest_similarity_score"] = lowest_distance
        metadata["similar_jailbreak_phrase"] = closest_chunk
        if lowest_distance < self._threshold:
            return FailResult(
                metadata=metadata,
                error_message=(
                        "The following text in your response is similar to our dataset of jailbreaks prompts:\n" + value
                ),
            )
        return PassResult(metadata=metadata)

    def query_vector_collection(
            self,
            text: str,
            k: int,
    ) -> List[Tuple[str, float]]:

        # Create embeddings
        query_embedding = self.embed_function(text).squeeze()

        # Compute distances
        cos_distances = 1 - (
                np.dot(self.source_embeddings, query_embedding)
                / (
                        np.linalg.norm(self.source_embeddings, axis=1)
                        * np.linalg.norm(query_embedding)
                )
        )
        
        # Sort indices from lowest cosine distance to highest distance
        low_to_high_ind = np.argsort(cos_distances)[:k]
        
        # Get top-k closest distances
        lowest_distances = [cos_distances[j] for j in low_to_high_ind]

        # Get top-k closest chunks
        closest_chunks = [self.chunks[j] for j in low_to_high_ind]

        return list(zip(closest_chunks, lowest_distances))
