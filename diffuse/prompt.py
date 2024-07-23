"""
Module: diffuse.prompt
"""

from typing import Optional

from tokenizers import AutoTokenizer


class PromptLengthError(ValueError):
    pass


def assert_prompt_length(
    tokenizer_path: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
) -> None:
    """
    Validate the length of a given prompt and optional negative prompt using
    a specified tokenizer. If either exceeds the model's maximum length limit,
    raise an error with suggestions for shortening prompts.

    Args:
        tokenizer_path (str): The path to the pre-trained tokenizer
        prompt (str): The main prompt used as input
        negative_prompt (Optional[str], optional): An optional negative
            prompt that can be provided to guide the model's output. Defaults
            to None if no negative prompt is needed.

    Raises:
        PromptLengthError: When either the main or negative prompts exceed
                           the tokenizer's maximum length limit

    Returns:
        None

    Examples:

        >>> assert_prompt_length("path/to/tokenizer", "A long prompt here.", "Another long negative prompt.")
        Prompt is okay: Using 50 tokens.
        Negative prompt is okay: Using 67 tokens.

        >>> assert_prompt_length("path/to/tokenizer", "This is a very very very very very very long prompt.", None)
        Traceback (most recent call last):
            ...
        PromptLengthError: Prompt is 201 out of 50 tokens. Shorten your prompts.
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    prompt_tokens = len(tokenizer.encode(prompt))
    negative_prompt_tokens = (
        len(tokenizer.encode(negative_prompt)) if negative_prompt else []
    )

    model_max_length = tokenizer.model_max_length

    if prompt_tokens > model_max_length:
        raise PromptLengthError(
            f"Prompt is {prompt_tokens} out of {model_max_length} tokens. "
            "Shorten your prompts."
        )

    print(f"Prompt is okay: Using {prompt_tokens} tokens.\n")

    if negative_prompt_tokens > model_max_length:
        raise PromptLengthError(
            f"Negative prompt is {negative_prompt_tokens} out of {model_max_length} tokens. Shorten your prompts."
        )

    print(f"Negative prompt is okay: Using {negative_prompt_tokens} tokens.\n")
