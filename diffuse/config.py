"""
Module: diffuse.config
"""

from typing import Any, Dict

import torch


def config_pipeline(
    **kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    config = {}
    config["variant"] = kwargs.get("variant", "fp16")
    config["torch_dtype"] = kwargs.get("torch_dtype", torch.bfloat16)
    config["use_safetensors"] = kwargs.get("use_safetensors", True)
    config["add_watermarker"] = kwargs.get("add_watermarker", True)
    config["use_single_file"] = kwargs.get("use_single_file", False)
    config["device"] = kwargs.get("device", "cpu")
    config.update(kwargs)
    return config
