"""
Module: diffuse.config
"""

from typing import Any, Dict

import torch


def config_pipeline(
    use_single_file: bool = False,
    use_safetensors: bool = True,
    device: str = "cpu",
    variant: str = "fp16",
    torch_dtype: torch.device = torch.bfloat16,
    **kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    config = {
        "use_single_file": use_single_file,
        "use_safetensors": use_safetensors,
        "device": device,
        "variant": variant,
        "torch_dtype": torch_dtype,
    }
    config.update(kwargs)
    return config
