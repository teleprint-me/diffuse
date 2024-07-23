"""
Script: diffuse.3.txt2img
"""

import argparse

import torch
from diffusers import StableDiffusion3Pipeline

from diffuse.prompt import assert_prompt_length


def initialize_pipeline(model_file_path: str, config: dict) -> StableDiffusion3Pipeline:
    # Create and configure the diffusion pipeline
    if config.get("use_single_file", False):
        pipe = StableDiffusion3Pipeline.from_single_file(
            model_file_path,
            **config,
        )
    else:  # kwargs not expected by StableDiffusionXLPipeline and are ignored
        for key in ["use_single_file"]:
            config.pop(key)
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_file_path,
            **config,
        )
    pipe.to(config.get("device", "cpu"))
    return pipe


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using diffusion-based models."
    )
    parser.add_argument("model", help="Path to the diffusion model file")
    parser.add_argument("prompt", help="Prompt for image generation")
    args = parser.parse_args()

    if args.tokenizer is not None:
        assert_prompt_length(args.tokenizer, args.prompt, args.negative_prompt)

    config = {
        "use_single_file": args.use_single_file,
        "device": args.device,
        "variant": "fp16",
        "torch_dtype": torch.bfloat16,
        "use_safetensors": True,
        "add_watermarker": False,
    }

    pipe = initialize_pipeline(args.model, config)

    image = pipe(
        "A cat holding a sign that says hello world",
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]


if __name__ == "__main__":
    main()
