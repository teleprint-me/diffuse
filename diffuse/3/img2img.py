"""
Script: diffuse.3.img2img
"""

import argparse

from diffusers import StableDiffusion3Img2ImgPipeline

from diffuse.config import config_pipeline
from diffuse.image import generate_image_to_image
from diffuse.pipeline import initialize_pipeline
from diffuse.prompt import assert_prompt_length


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images using images with Stable Diffusion 3 models."
    )
    parser.add_argument("model", help="Path to the diffusion model or model ID")
    parser.add_argument("image", help="Path to the initial image")
    parser.add_argument("prompt", help="Prompt for image generation")
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
        "--delay",
        type=float,
        default=1 / 30,
        help="Delay between image generation. Default is (float) 0.0333...",
    )
    parser.add_argument(
        "--device", default="cpu", help="The device to use. Defaults to 'cpu'."
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to the models tokenizer. Defaults to None.",
    )
    parser.add_argument(
        "--add_watermarker",
        action="store_false",
        help="Watermark generated images. Defaults to True.",
    )
    return parser.parse_args()


def main():
    args = get_arguments()

    if args.tokenizer is not None:
        assert_prompt_length(args.tokenizer, args.prompt, args.negative_prompt)

    config = config_pipeline(device=args.device, add_watermarker=args.add_watermarker)
    pipe_image = initialize_pipeline(
        args.model,
        StableDiffusion3Img2ImgPipeline,
        config,
    )

    images, elapsed_time = generate_image_to_image(
        pipe_image=pipe_image,
        image_path=args.image,
        prompt=args.prompt,
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
