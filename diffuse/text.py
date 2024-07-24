"""
Module: diffuse.text
"""

import os
from datetime import datetime
from time import sleep
from typing import List, Optional, Tuple

import numpy as np
from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from PIL import Image


def handle_pipeline_result(result: StableDiffusionPipelineOutput) -> List[Image]:
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


def write_images(
    images: List[Image],
    output_directory: Optional[str] = "images",
    delay: Optional[float] = 1 / 30,
) -> List[Tuple[Image, str]]:
    """
    Write the generated images to the specified output directory.

    Args:
        images (List[Image]): List of generated images.
        output_directory (str, optional): Directory to save generated images.
        delay (float, optional): Delay between image generation.

    Returns:
        List[Tuple[Image, str]]: List of tuples containing generated images and their paths.
    """

    dataset = []

    for image in images:
        image_path = f"{output_directory}/{datetime.now()}.png"
        image.save(image_path)
        dataset.append((image, image_path))
        print(f"Created: {image_path}")
        sleep(delay)  # NOTE: Prevent overwrites

    return dataset


def generate_text_to_image(
    pipe_text: DiffusionPipeline,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7,
    num_images_per_prompt: int = 2,
    output_directory: str = "images",
    delay: float = 1 / 30,
) -> tuple[list[tuple[Image, str]], float]:
    try:
        # Generate images based on the provided prompts
        start_time = datetime.now()

        os.makedirs(output_directory, exist_ok=True)

        result = pipe_text(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        )

        # Handle different types of results (list or numpy array)
        images = handle_pipeline_result(result)
        # Create a unique filename using the current timestamp
        dataset = write_images(images, output_directory, delay)

        # Calculate the elapsed time
        end_time = datetime.now()
        elapsed_time = end_time - start_time

        return dataset, elapsed_time
    except KeyboardInterrupt:
        # NOTE: Gracefully interrupt image generation
        print("KeyboardInterrupt: Exiting now.")
        exit(1)
