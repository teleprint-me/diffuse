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

from diffuse.generate import generate_text_to_image
from diffuse.pipeline import pipeline_config, pipeline_initialize
from diffuse.prompt import assert_prompt_length


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images using text with stable-diffusion-xl models."
    )
    parser.add_argument("model_path", help="Path to the diffusion model or model ID")
    parser.add_argument("prompt", help="Prompt for image generation")
    parser.add_argument(
        "--negative_prompt", help="Negative prompt for image generation"
    )
    parser.add_argument(
        "--output_dir", default="images", help="Directory to save generated images"
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
        type=float,
        default=7,
        help="Guidance scale for image generation. Default is (float) 7.0.",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_false",
        help="Use safetensors. Default is True.",
    )
    parser.add_argument(
        "--add_watermarker",
        action="store_true",
        help="Apply watermarker to images. Default is False.",
    )
    parser.add_argument(
        "--use_single_file",
        action="store_true",
        help="Use a fine-tuned model. Default is False.",
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
        help="Delay between image generation. Default is (float) 1/30.",
    )
    parser.add_argument(
        "--device_type",
        default="cpu",
        help="The device to use. Defaults to 'cpu'.",
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

    config = pipeline_config(
        variant="fp16",
        torch_dtype=torch.bfloat16,
        use_safetensors=args.use_safetensors,
        use_single_file=args.use_single_file,
        # NOTE: diffusers tests for None for add_watermarker
        # it does not test for a bool even though it expects a bool
        # this seems more like a semantic mistake that can trigger odd bugs
        # setting this flag seems to trigger a continuous loop that only ends
        # if the user interrupts the program. this is not ideal and constitutes
        # a complete failure in the use and implementation of this feature.
        add_watermarker=args.add_watermarker,
    )

    pipe_text = pipeline_initialize(
        model_file_path=args.model_path,
        pipeline_class=StableDiffusionXLPipeline,
        pipeline_config=config,
        device_type=args.device_type,  # 'cpu', 'cuda', etc.
    )

    if args.lora_path is not None:
        pipe_text.load_lora_weights(args.lora_path, adapter_name=args.adapter_name)

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
