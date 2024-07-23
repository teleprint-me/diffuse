"""
Module: diffuse.pipeline
"""

from diffusers.pipelines import DiffusionPipeline


def initialize_pipeline(
    model_file_path: str,
    config: dict,
    pipeline: DiffusionPipeline,
) -> DiffusionPipeline:
    # Create and configure the diffusion pipeline
    if config.get("use_single_file", False):
        pipe = pipeline.from_single_file(
            model_file_path,
            **config,
        )
    else:  # kwargs not expected by the pipeline and are ignored
        for key in ["use_single_file"]:
            config.pop(key)
        pipe = pipeline.from_pretrained(
            model_file_path,
            **config,
        )
    pipe.to(config.get("device", "cpu"))
    return pipe
