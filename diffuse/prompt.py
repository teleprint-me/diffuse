"""
Module: diffuse.prompt
"""

from typing import Optional

from tokenizers import AutoTokenizer


def assert_prompt_length(
    tokenizer_path: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompt_tokens = len(tokenizer.encode(prompt))
    negative_prompt_tokens = len(
        tokenizer.encode(negative_prompt) if negative_prompt else []
    )
    model_max_length = tokenizer.model_max_length
    if prompt_tokens > model_max_length:
        raise ValueError(
            f"Prompt is {prompt_tokens} out of {model_max_length} tokens. Shorten your prompts."
        )
    print(f"Prompt is okay: Using {prompt_tokens} tokens.\n")
    if negative_prompt_tokens > tokenizer.model_max_length:
        raise ValueError(
            f"Negative prompt is {negative_prompt_tokens} out of {model_max_length} tokens. Shorten your prompts."
        )
    print(f"Negative prompt is okay: Using {negative_prompt_tokens} tokens.\n")
