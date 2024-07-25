#!/usr/bin/env python

"""
Script: diffuse.xl.txt2img

Documentation:
- https://huggingface.co/stabilityai/sdxl-turbo
- https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTokenizer
- https://huggingface.co/docs/diffusers/main/en/api/loaders/lora
- https://huggingface.co/guoyww/animatediff/tree/main
"""

import argparse

import torch
from diffusers import StableDiffusionXLPipeline

from diffuse.pipeline import initialize_pipeline
from diffuse.prompt import assert_prompt_length
from diffuse.text import generate_text_to_image


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images using text with stable-diffusion-xl models."
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
        "--tokenizer", default=None, help="Path to the models tokenizer"
    )
    parser.add_argument(
        "--output_dir", default="images", help="Directory to save generated images"
    )
    parser.add_argument(
        "--num_images", type=int, default=2, help="Number of images to generate"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=int,
        default=7,
        help="Guidance scale for image generation",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights",
    )
    parser.add_argument(
        "--adapter_name",
        default=None,
        help="Name of the LoRA adapter",
    )
    parser.add_argument(
        "--timer", type=float, default=1 / 30, help="Delay between image generation"
    )
    parser.add_argument(
        "--device", default="cpu", help="The device to use. Defaults to 'cpu'"
    )
    return parser.parse_args()


def main():
    args = get_arguments()

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

    pipe = initialize_pipeline(args.model, StableDiffusionXLPipeline, config)

    if args.lora_path is not None:
        pipe.load_lora_weights(args.lora_path, adapter_name=args.adapter_name)

    images, elapsed_time = generate_text_to_image(
        pipe,
        args.prompt,
        args.negative_prompt,
        args.num_steps,
        args.guidance_scale,
        args.num_images,
        args.output_dir,
        args.timer,
    )

    print(f"Elapsed time: {elapsed_time}")


if __name__ == "__main__":
    main()
