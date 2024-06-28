import itertools
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import pandas as pd
import os

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
    """Function used to embed text with OpenAIEmbedding(model="text-embedding-ada-002").

    :param text: Either a string or list of strings that will be embedded.

    :return: Array of embedded input string(s).
    """
    if isinstance(text, str):
        text = [text]

    embeddings_out = []
    for current_example in text:
        embeddings_out.append(OpenAIEmbedding(model="text-embedding-ada-002").get_text_embedding(current_example))
    return np.array(embeddings_out)


def get_prompts(filename: str) -> List[str]:
    """Get prompts from local file.
    
    :param filename: Name of CSV file (excluding directory), e.g. my_file.csv.

    :return: List of prompt strings, e.g. ["my prompt 1", "my prompt 2", ..., "my prompt 500"]
    """
    script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
    # Dataset from public repo associated with arxiv paper https://github.com/verazuo/jailbreak_llms
    file_path = os.path.join(script_dir, filename)
    prompts = pd.read_csv(file_path)["prompt"].tolist()
    return prompts


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
        threshold: float = 0.2,
        validation_method: str = "full",
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            on_fail, threshold=threshold, validation_method=validation_method, **kwargs
        )
        self._threshold = float(threshold)
        self._validation_method = "full"
        self.sources = kwargs.get("sources", None)
        
        # Use Arize AI prompts if user does not provide their own.
        if self.sources is None:
            script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
            # Dataset from public repo associated with arxiv paper https://github.com/verazuo/jailbreak_llms
            file_path = os.path.join(script_dir, 'jailbreak_prompts_2023_05_07.csv')
            # We recommend at least 10 examples. Additional examples may adversely affect latency.
            self.sources = pd.read_csv(file_path)["prompt"].tolist()[:10]

        # Validate user inputs
        for prompt in self.sources:
            if not prompt or not isinstance(prompt, str):
                raise ValueError(f"Prompt example: {prompt} is invalid. Must contain valid string data.")

        self.embed_function = kwargs.get("embed_function", _embed_function)

        # Check chunking strategy
        chunk_strategy = kwargs.get("chunk_strategy",EmbeddingChunkStrategy.SENTENCE.name.lower())
        chunk_size = kwargs.get("chunk_size", 100)
        chunk_overlap = kwargs.get("chunk_overlap", 20)

        chunks = [
            get_chunks_from_text(source, chunk_strategy, chunk_size, chunk_overlap)
            for source in self.sources
        ]
        self.chunks = list(itertools.chain.from_iterable(chunks))

        # Create embeddings
        self.source_embeddings = np.array(self.embed_function(self.chunks)).squeeze()

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        """Validation function for the JailbreakEmbeddings validator. If the cosine distance
        of the user input embeddings is below the user-specified threshold of the closest embedded chunk
        from the jailbreak examples in prompt sources, then the Guard will return FailResult. If all chunks
        are sufficiently distant, then the Guard will return PassResult.

        :param value: This is the 'value' of user input. For the JailbreakEmbeddings Guard, we want
            to ensure we are validating user input, rather than LLM output, so we need to call
            the guard with Guard().use(JailbreakEmbeddings, on="prompt")

        :return: PassResult or FailResult.
        """
        closest_chunk, lowest_distance = self.query_vector_collection(text=metadata.get("user_input"), k=1)[0]
        metadata["highest_similarity_score"] = lowest_distance
        metadata["similar_jailbreak_phrase"] = closest_chunk
        if lowest_distance < self._threshold:
            # At least one jailbreak embedding chunk was within the cosine distance threshold from the user input embedding
            return FailResult(
                metadata=metadata,
                error_message=(
                    f"The following text in your response is similar to our dataset of jailbreaks prompts:\n{value}"
                ),
                fix_value="I'm sorry, I cannot respond. The Arize Guard flagged this message as a Jailbreak attempt."
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
