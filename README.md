# Overview

| Developed by | Arize AI |
| Date of development | July 2, 2024 |
| Validator type | Dataset, Embeddings |
| Blog | TODO(Julia) Add link to docs |
| License | Apache 2 |
| Input/Output | Input or Output |

## Description

Provided an input dataset of "bad" prompts or LLM messages, this Guard will block the LLM from responding to similar prompts or producing similar LLM messages. First, the Guard computes embeddings over split chunks of the few shot examples in a source dataset (we recommend 10 examples) and stores them upon construction. When the Guard is later called on an input message, the Guard checks the cosine distance between the embedded incoming message and the embedded chunks from the source dataset. If any of the chunks are within a user-specified threshold (defaults to 0.2), then the Guard will take the user-specified action to block the LLM call.

This is a very flexible Guard. In our demo notebook, the Guard is separately applied to a public dataset of Jailbreak prompts and a public dataset of PII inputs. This Guard can be updated easily by changing the dataset to accommodate specialized use cases or evolving prompt injection techniques.

### Intended Use
This validator is a template for creating other validators, but for demonstrative purposes it ensures that a generated output is the literal `pass`.

### Benchmark results
We benchmarked the Arize JailbreakEmbeddings Guard on 656 prompts contained in the dataset validator/jailbreak_prompts_2023_05_07.csv.
The benchmarking code is provided in validator/benchmark_guard_on_dataset.py.
For details on the dataset, please refer to the following resources:
* Research Paper on Arxiv: https://arxiv.org/pdf/2308.03825
* Repository containing the dataset and other benchmarks: https://github.com/verazuo/jailbreak_llms
* Website associated with original research paper: https://jailbreak-llms.xinyueshen.me/
* URL link to original dataset: https://github.com/verazuo/jailbreak_llms/tree/main/data

The Arize JailbreakEmbeddings Guard has the following results
86.43% of 656 jailbreak prompts failed the JailbreakEmbeddings guard.
13.57% of 656 jailbreak prompts passed the JailbreakEmbeddings guard.
567 True Positives
89 False Negatives
1.41 median latency
2.91 mean latency
618.54 max latency

### Requirements

* Dependencies:
	- guardrails-ai>=0.4.0

* Foundation model access keys:
	- OPENAI_API_KEY

## Installation

```bash
$ guardrails hub install hub://guardrails/validator_template
```

## Usage Examples

### Validating string output via Python

In this example, we apply the validator to a string output generated by an LLM.

```python
# Import Guard and Validator
from guardrails.hub import ValidatorTemplate
from guardrails import Guard

# Setup Guard
guard = Guard().use(
    ValidatorTemplate
)

guard.validate("pass")  # Validator passes
guard.validate("fail")  # Validator fails
```

### Validating JSON output via Python

In this example, we apply the validator to a string field of a JSON output generated by an LLM.

```python
# Import Guard and Validator
from pydantic import BaseModel, Field
from guardrails.hub import ValidatorTemplate
from guardrails import Guard

# Initialize Validator
val = ValidatorTemplate()

# Create Pydantic BaseModel
class Process(BaseModel):
		process_name: str
		status: str = Field(validators=[val])

# Create a Guard to check for valid Pydantic output
guard = Guard.from_pydantic(output_class=Process)

# Run LLM output generating JSON through guard
guard.parse("""
{
	"process_name": "templating",
	"status": "pass"
}
""")
```

# API Reference

**`__init__(self, on_fail="noop")`**
<ul>
Initializes a new instance of the ValidatorTemplate class.

**Parameters**
- **`arg_1`** *(str)*: A placeholder argument to demonstrate how to use init arguments.
- **`arg_2`** *(str)*: Another placeholder argument to demonstrate how to use init arguments.
- **`on_fail`** *(str, Callable)*: The policy to enact when a validator fails.  If `str`, must be one of `reask`, `fix`, `filter`, `refrain`, `noop`, `exception` or `fix_reask`. Otherwise, must be a function that is called when the validator fails.
</ul>
<br/>

**`validate(self, value, metadata) -> ValidationResult`**
<ul>
Validates the given `value` using the rules defined in this validator, relying on the `metadata` provided to customize the validation process. This method is automatically invoked by `guard.parse(...)`, ensuring the validation logic is applied to the input data.

Note:

1. This method should not be called directly by the user. Instead, invoke `guard.parse(...)` where this method will be called internally for each associated Validator.
2. When invoking `guard.parse(...)`, ensure to pass the appropriate `metadata` dictionary that includes keys and values required by this validator. If `guard` is associated with multiple validators, combine all necessary metadata into a single dictionary.

**Parameters**
- **`value`** *(Any)*: The input value to validate.
- **`metadata`** *(dict)*: A dictionary containing metadata required for validation. Keys and values must match the expectations of this validator.
    
    
    | Key | Type | Description | Default |
    | --- | --- | --- | --- |
    | `key1` | String | Description of key1's role. | N/A |
</ul>
