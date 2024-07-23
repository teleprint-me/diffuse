# https://huggingface.co/stabilityai/sdxl-turbo
# https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTokenizer
# https://huggingface.co/docs/diffusers/main/en/api/loaders/lora
# https://huggingface.co/guoyww/animatediff/tree/main
import argparse
from datetime import datetime
from time import sleep

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from diffuse.pipeline import initialize_pipeline
from diffuse.prompt import assert_prompt_length


def generate_images(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7,
    num_images_per_prompt: int = 2,
    output_directory: str = "images",
    timer: float = 1 / 30,
) -> tuple[list[tuple[Image, str]], float]:
    # Generate images based on the provided prompts
    dataset = []
    start_time = datetime.now()

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        )

        # Handle different types of results (list or numpy array)
        if isinstance(result.images, list):
            images = result.images
        elif isinstance(result.images, np.ndarray):
            images = [Image.fromarray(img) for img in result.images]
        else:
            raise ValueError("Unsupported image format")

        # Create a unique filename using the current timestamp
        for image in images:
            image_path = f"{output_directory}/{datetime.now()}.png"
            image.save(image_path)
            dataset.append((image, image_path))
            print(f"Created: {image_path}")
            sleep(timer)  # NOTE: Prevent overwrites
    except KeyboardInterrupt:
        # NOTE: Gracefully interrupt image generation
        print("KeyboardInterrupt: Exiting now.")
        exit(1)

    end_time = datetime.now()
    elapsed_time = end_time - start_time

    return dataset, elapsed_time


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using diffusion-based models."
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

    pipe = initialize_pipeline(args.model, StableDiffusionXLPipeline, config)

    if args.lora_path is not None:
        pipe.load_lora_weights(args.lora_path, adapter_name=args.adapter_name)

    images, elapsed_time = generate_images(
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
