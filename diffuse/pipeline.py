"""
Module: diffuse.pipeline

Provides functions for initializing diffusion pipelines based on a given
class (e.g., StableDiffusion3Pipeline) and configuration options, such as
the device to use during computations or whether to load the pipeline from
a single pre-trained file.
"""

from typing import Any, Dict, List, Optional, Type

from diffusers.loaders import FromSingleFileMixin
from diffusers.pipelines import DiffusionPipeline


def initialize_pipeline(
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
        pipeline_class (type[DiffusionPipeline]): The Diffusers diffusion
            pipeline class to utilize
        pipeline_config (dict): Configuration options for initializing the pipeline,
            such as device or whether to load from a single pre-trained
            file.

    Returns:
        DiffusionPipeline: The initialized diffusion pipeline

    Raises:
        FileNotFoundError: If the specified model_file_path is invalid

    Examples:

        >>> initialize_pipeline("my_model_directory", StableDiffusion3Pipeline, {"device": "cpu"})

    """
    pipe = None

    if pipeline_config is None:
        pipeline_config = {}

    if filter_pipeline_kwargs is None:
        filter_pipeline_kwargs = ["use_single_file"]

    if device_type is None:
        device_type = "cpu"

    # from_single_file method is only available to classes inheriting from FromSingleFileMixin
    if pipeline_config.get("use_single_file") is True:
        # if isinstance(pipeline_class, FromSingleFileMixin):
        # NOTE: I think this doesn't work because pipeline is not a class instance
        # this means it it did not inherit from its parent classes
        # and is not a child class of FromSingleFileMixin as a consequence of this.
        # this is why pipe.to(device_type) ends up being an exception of a NoneType object
        # this matters because the from_single_file is only available if the child class
        # inherits from FromSingleFileMixin, but this doesn't seem to be an issue here
        # and, strangely enough, works regardless.
        pipe = pipeline_class.from_single_file(model_file_path, **pipeline_config)
    else:  # kwargs not expected by the pipeline and are ignored
        for key in filter_pipeline_kwargs:
            pipeline_config.pop(key)
        pipe = pipeline_class.from_pretrained(model_file_path, **pipeline_config)

    pipe.to(device_type)

    return pipe
