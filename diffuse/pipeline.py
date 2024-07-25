"""
Module: diffuse.pipeline

Provides functions for initializing diffusion pipelines based on a given
class (e.g., StableDiffusion3Pipeline) and configuration options, such as
the device to use during computations or whether to load the pipeline from
a single pre-trained file.
"""

from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from diffusers.loaders import FromSingleFileMixin
from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from PIL import Image


def pipeline_config(
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


def pipeline_initialize(
    model_file_path: str,
    pipeline_class: Type[DiffusionPipeline],
    pipeline_config: Optional[Dict[str, Any]] = None,
    filter_pipeline_kwargs: Optional[List[str]] = None,
    device_type: Optional[str] = None,
) -> DiffusionPipeline:
    """
    Initialize the diffusion pipeline for a given class (e.g.,
    StableDiffusion3Pipeline or StableDiffusionXLPipeline).

    Args:
        model_file_path (str): The path to the pre-trained model file
        pipeline_class (Type[DiffusionPipeline]): The Diffusers diffusion
            pipeline class to utilize
        pipeline_config (Optional[Dict[str, Any]]): Configuration options for initializing the pipeline,
            such as device or whether to load from a single pre-trained file.
        filter_pipeline_kwargs (Optional[List[str]]): List of keywords to filter out from pipeline configuration.
        device_type (Optional[str]): The device type to use (e.g., 'cpu' or 'cuda').

    Returns:
        DiffusionPipeline: The initialized diffusion pipeline

    Raises:
        FileNotFoundError: If the specified model_file_path is invalid

    Examples:
        >>> initialize_pipeline("my_model_directory", StableDiffusion3Pipeline, {"device": "cpu"})

    """
    if pipeline_config is None:
        pipeline_config = {}

    if filter_pipeline_kwargs is None:
        filter_pipeline_kwargs = ["use_single_file"]

    if device_type is None:
        device_type = "cpu"

    pipe = None

    # from_single_file method is only available to classes inheriting from FromSingleFileMixin
    if pipeline_config.get("use_single_file") is True:
        if issubclass(pipeline_class, FromSingleFileMixin):
            pipe = pipeline_class.from_single_file(model_file_path, **pipeline_config)
        else:
            raise TypeError(
                f"{pipeline_class.__name__} does not support loading from a single file."
            )
    else:  # kwargs not expected by the pipeline and are ignored
        for key in filter_pipeline_kwargs:
            pipeline_config.pop(key, None)
        pipe = pipeline_class.from_pretrained(model_file_path, **pipeline_config)

    pipe.to(device_type)

    return pipe


def pipeline_handle_result(result: StableDiffusionPipelineOutput) -> List[Image]:
    """
    Handle the result from the diffusion pipeline and convert it into a list of PIL Images.

    Args:
        result (StableDiffusionPipelineOutput): Result from the diffusion pipeline.

    Returns:
        List[Image]: List of generated images.
    """

    if isinstance(result.images, list):
        images = result.images
    elif isinstance(result.images, np.ndarray):
        images = [Image.fromarray(img) for img in result.images]
    else:
        raise ValueError("Unsupported image format")

    return images
