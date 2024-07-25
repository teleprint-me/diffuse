#!/usr/bin/env python

"""
Script: diffuse.xl.img2img

Documentation:
- https://huggingface.co/stabilityai/sdxl-turbo
- https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTokenizer
- https://huggingface.co/docs/diffusers/main/en/api/loaders/lora
- https://huggingface.co/guoyww/animatediff/tree/main
"""

import argparse

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

from diffuse.config import config_pipeline
from diffuse.image import generate_image_to_image
from diffuse.pipeline import initialize_pipeline
from diffuse.prompt import assert_prompt_length


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images using images with stable-diffusion-xl models."
    )
    parser.add_argument("model_path", help="Path to the diffusion model or model ID")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("prompt", help="Prompt for image generation")
    parser.add_argument(
        "--use_single_file", action="store_true", help="Use a fine-tuned model"
    )
    parser.add_argument(
        "--negative_prompt", help="Negative prompt for image generation"
    )
    parser.add_argument(
        "--output_dir", default="images", help="Directory to save generated images"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.5,
        help="The amount of noise added to the image. Values must be between 0 and 1. Default is (float) 0.5.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7,
        help="Guidance scale for image generation. Default is (float) 7.0.",
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
        help="Name of the LoRA adapter. (Optional) Name for the adapter.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1 / 30,
        help="Delay between image generation. Default is (float) 0.0333...",
    )
    parser.add_argument(
        "--device-type", default="cpu", help="The device to use. Defaults to 'cpu'."
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to the models tokenizer. Defaults to None.",
    )
    return parser.parse_args()


def main():
    args = get_arguments()

    if args.tokenizer is not None:
        assert_prompt_length(args.tokenizer, args.prompt, args.negative_prompt)

    # variable keyword arguments to pass to the pipeline for instantiation
    config = config_pipeline(
        variant="fp16",
        use_safetensors=True,
        add_watermarker=False,
        torch_dtype=torch.bfloat16,
        use_single_file=args.use_single_file,
    )

    pipe = initialize_pipeline(
        model_file_path=args.model_path,
        pipeline_class=StableDiffusionXLImg2ImgPipeline,
        pipeline_config=config,
        device_type=args.device_type,  # 'cpu', 'cude', etc.
    )

    if args.lora is True:
        pipe.load_lora_weights(args.lora_path, args.adapter_name)

    images, elapsed_time = generate_image_to_image(
        pipe,
        args.image_path,
        args.prompt,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        output_directory=args.output_dir,
        delay=args.delay,
    )

    print(f"Elapsed time: {elapsed_time}")


if __name__ == "__main__":
    main()
