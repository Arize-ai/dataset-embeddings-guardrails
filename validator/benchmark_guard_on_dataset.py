"""Benchmark Arize ArizeDatasetEmbeddings Guard against a dataset of regular prompts and a dataset of jailbreak prompts."""
import os
from getpass import getpass
from typing import List
import time
import statistics

import pandas as pd
import openai
from sklearn.utils import shuffle

from typing import List, Optional
import time
import statistics

from guardrails import Guard
from guardrails.llm_providers import PromptCallableException
import openai

from main import ArizeDatasetEmbeddings


JAILBREAK_PROMPTS_FP = "jailbreak_prompts_2023_05_07.csv"
VANILLA_PROMPTS_FP = "regular_prompts_2023_05_07.csv"
# Number of few-shot examples to show Guard what a jailbreak prompts looks like.
# Too few examples will result in False Negatives, while too many examples will result in worse latency.
NUM_FEW_SHOT_EXAMPLES = 10
MODEL = "gpt-3.5-turbo"
# Output file to log debugging info. Set to None if you do not wish to add logging.
OUTFILE = f"arize_{ArizeDatasetEmbeddings.__name__}_guard_{MODEL}_output.txt"


def append_to_file(filepath: str, text: str) -> None:
    """Append debugging text to output filepath.

    :param filepath: String defining the filepath of output text.
    :param text: String message to append to the filepath.
    """
    with open(filepath, "a") as f:
        f.write(f"\nprompt:\n{text}")


def evaluate_embeddings_guard_on_dataset(test_prompts: List[str], guard: Guard, outfile: Optional[str]):
    """Evaluate Embeddings guard on benchmark dataset. Refer to README for details and links to arxiv paper. This
    will calculate the number of True Positives, False Positives, True Negatives and False Negatives.

    :param test_prompts: List of string prompts where we want to evaluate the Guard's response.
    :param guard: Guard we want to evaluate.
    :param train_prompts: List of source prompts that serve as a few-shot guide to prompts we want to guard against.
    :param outfile: Output file for debugging information. If None, then we do not write logging information to a file.
    """
    latency_measurements = []
    num_passed_guard = 0
    num_failed_guard = 0
    for prompt in test_prompts:
        try:
            start_time = time.perf_counter()
            response = guard(
                llm_api=openai.chat.completions.create,
                prompt=prompt,
                model="gpt-3.5-turbo",
                max_tokens=1024,
                temperature=0.5,
                metadata={
                    "user_message": prompt,
                }
            )
            latency_measurements.append(time.perf_counter() - start_time)
            if response.validation_passed:
                num_passed_guard += 1
            else:
                num_failed_guard += 1
            total = num_passed_guard + num_failed_guard
            if outfile is not None:
                debug_text = f"""\nprompt:\n{prompt}\nresponse:\n{response}\n{100 * num_failed_guard / total:.2f}% of {total} 
                    prompts failed the ArizeDatasetEmbeddings guard.\n{100 * num_passed_guard / total:.2f}% of {total} prompts 
                    passed the ArizeDatasetEmbeddings guard."""
                append_to_file(filepath=outfile, text=debug_text)
        except PromptCallableException as e:
            # Dataset may contain a few bad apples that result in an Open AI error for invalid inputs.
            # Catch and log the exception, then continue benchmarking the valid examples.
            append_to_file(filepath=outfile, text=f"\nexception:\n{e}")
    return num_passed_guard, num_failed_guard, latency_measurements


def benchmark_arize_jailbreak_embeddings_validator(train_prompts: List[str], jailbreak_test_prompts: List[str], vanilla_prompts: List[str], outfile: Optional[str]):
    """Benchmark Arize ArizeDatasetEmbeddings Guard against a dataset of regular prompts and a dataset of jailbreak prompts.

    :param train_prompts: Few-shot examples of jailbreak prompts.
    :param jailbreak_test_prompts: Test prompts used to evaluate the Guard. We expect the Guard to block these examples.
    :param vanilla_prompts: Regular prompts used to evaluate the Guard. We expect the Guard to pass these examples.
    :param outfile: Filepath where we output debugging information, including TP, FP, TN, FN, latency per call, aggregate latency statistics,
        original prompt and validator response.
    """
    # Set up Guard
    guard = Guard.from_string(
        validators=[
            ArizeDatasetEmbeddings(threshold=0.2, validation_method="full", on_fail="refrain", sources=train_prompts)
        ],
    )
    
    # Evaluate Guard on dataset of jailbreak prompts
    num_passed_guard, num_failed_guard, latency_measurements = evaluate_embeddings_guard_on_dataset(
        test_prompts=jailbreak_test_prompts,
        guard=guard, outfile=outfile)
    if outfile is not None:
        debug_text = f"""\n{num_failed_guard} True Positives.\n{num_passed_guard} False Negatives. \n{statistics.median(latency_measurements)}
            median latency\n{statistics.mean(latency_measurements)} mean latency\n{max(latency_measurements)} max latency"""
        append_to_file(filepath=OUTFILE, text=debug_text)
    
    # Evaluate Guard on dataset of regular prompts
    num_passed_guard, num_failed_guard, latency_measurements = evaluate_embeddings_guard_on_dataset(
        test_prompts=vanilla_prompts,
        guard=guard, outfile=outfile)
    if outfile is not None:
        debug_text = f"""\n{num_failed_guard} True Negatives\n{num_passed_guard} False Positives \n{statistics.median(latency_measurements)}
            median latency\n{statistics.mean(latency_measurements)} mean latency\n{max(latency_measurements)} max latency"""
        append_to_file(filepath=OUTFILE, text=debug_text)


def get_prompts(filename: str) -> List[str]:
    script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
    # Dataset from public repo associated with arxiv paper https://github.com/verazuo/jailbreak_llms
    file_path = os.path.join(script_dir, filename)
    prompts = shuffle(pd.read_csv(file_path))["prompt"].tolist()
    return prompts


def main():
    # Jailbreak prompts that we expect to Fail the Guard (656 examples)
    jailbreak_prompts = get_prompts(JAILBREAK_PROMPTS_FP)
    train_prompts = jailbreak_prompts[-NUM_FEW_SHOT_EXAMPLES:]
    test_prompts = jailbreak_prompts[:-NUM_FEW_SHOT_EXAMPLES]

    # Vanilla prompts that we expect to Pass the Guard
    vanilla_prompts = get_prompts(VANILLA_PROMPTS_FP)

    benchmark_arize_jailbreak_embeddings_validator(
        jailbreak_test_prompts=test_prompts,
        vanilla_prompts=vanilla_prompts,
        train_prompts=train_prompts,
        outfile=OUTFILE)


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = getpass("🔑 Enter your OpenAI API key: ")
    main()
