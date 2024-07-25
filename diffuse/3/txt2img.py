#!/usr/bin/env python

"""
Script: diffuse.3.txt2img

This script demonstrates how to generate images using the Stable Diffusion 3 model
from the Diffusers library, given a prompt and configuration options such as
guidance scale and number of inference steps.

The generated image is then saved as an PNG file for easy viewing.
"""

import argparse

from diffusers import StableDiffusion3Pipeline

from diffuse.config import config_pipeline
from diffuse.pipeline import initialize_pipeline
from diffuse.prompt import assert_prompt_length
from diffuse.text import generate_text_to_image


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images using text with Stable Diffusion 3 models."
    )
    parser.add_argument("model", help="Path to the diffusion model file")
    parser.add_argument("prompt", help="Prompt for image generation")
    parser.add_argument(
        "--use_single_file", action="store_true", help="Use a fine-tuned model"
    )
    parser.add_argument(
        "--negative_prompt", help="Negative prompt for image generation"
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to the models tokenizer. Default is None.",
    )
    parser.add_argument(
        "--output_dir",
        default="images",
        help="Directory to save generated images. Default is 'images'.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=2,
        help="Number of images to generate. Default is (int) 2.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of inference steps. Default is (int) 50.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=int,
        default=7,
        help="Guidance scale for image generation. Default is (float) 7.0f.",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA for fine-tuning the model. Default is False.",
    )
    parser.add_argument(
        "--lora_path",
        help="Path to LoRA weights. --lora is required to execute.",
    )
    parser.add_argument(
        "--adapter_name",
        default=None,
        help="Name of the LoRA adapter. (Optional) Default is None.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1 / 30,
        help="Delay between image generation. Default is (float) 0.0333...",
    )
    parser.add_argument(
        "--device", default="cpu", help="The device to use. Defaults to 'cpu'"
    )
    return parser.parse_args()


def main():
    args = get_arguments()

    if args.tokenizer is not None:
        assert_prompt_length(args.tokenizer, args.prompt, args.negative_prompt)

    config = config_pipeline(device=args.device, use_single_file=args.use_single_file)
    pipe_text = initialize_pipeline(args.model, StableDiffusion3Pipeline, config)

    if args.lora is True:
        pipe_text.load_lora_weights(args.lora_path, args.adapter_name)

    images, elapsed_time = generate_text_to_image(
        pipe_text=pipe_text,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        output_directory=args.output_dir,
        delay=args.delay,
    )

    print(f"Elapsed time: {elapsed_time}")


if __name__ == "__main__":
    main()
