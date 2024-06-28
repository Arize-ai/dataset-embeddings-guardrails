# to run these, run 
# pytest tests/test_validator.py
import pandas as pd
import os

from typing import List
from guardrails import Guard
import pytest
from validator import JailbreakEmbeddings

# We use 'exception' as the validator's fail action,
#  so we expect failures to always raise an Exception
# Learn more about corrective actions here:
#  https://www.guardrailsai.com/docs/concepts/output/#%EF%B8%8F-specifying-corrective-actions
guard = Guard().use(validator=JailbreakEmbeddings, on="prompt", on_fail="exception")

def get_prompts(filename: str) -> List[str]:
  script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
  # Dataset from public repo associated with arxiv paper https://github.com/verazuo/jailbreak_llms
  file_path = os.path.join(script_dir, filename)
  prompts = pd.read_csv(file_path)["prompt"].tolist()
  return prompts
   

def test_pass():
  vanilla_prompts = get_prompts("regular_prompts_2023_05_07.csv")[-2:]
  print(vanilla_prompts)
  for prompt in vanilla_prompts:
    guard.parse(prompt)
    assert guard.validation_passed == True


def test_fail():
  jailbreak_prompts = get_prompts('jailbreak_prompts_2023_05_07.txt')[-2:]
  for prompt in jailbreak_prompts:
    with pytest.raises(Exception) as exc_info:
      guard.parse(prompt)
  
  # Assert the exception has your error_message
  assert str(exc_info.value) == "Validation failed for field with errors: {A descriptive but concise error message about why validation failed}"
