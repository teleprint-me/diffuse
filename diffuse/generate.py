"""
Module: diffuse.generate
"""

from datetime import datetime
from time import sleep
from typing import List, Optional, Tuple

import numpy as np
from diffusers.pipelines import DiffusionPipeline
from PIL import Image


def generate_text_to_image(
    pipe_text: DiffusionPipeline,
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
        result = pipe_text(
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


def generate_image_to_image(
    pipe_image: DiffusionPipeline,
    image_path: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    strength=0.75,
    num_inference_steps: int = 50,
    guidance_scale=7.5,
    num_images_per_prompt: int = 2,
    output_directory: str = "images",
    delay: float = 1 / 30,
    dimensions: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Tuple[Image, str]], float]:
    try:
        start_time = datetime.now()
        init_image = initialize_image(image_path, dimensions)

        adjusted_inference_steps = adjust_inference_steps(strength, num_inference_steps)

        result = pipe_image(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=adjusted_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
        )

        images = handle_pipeline_result(result)
        dataset = write_images(images, output_directory, delay)

        end_time = datetime.now()
        elapsed_time = end_time - start_time

        return dataset, elapsed_time

    except KeyboardInterrupt:
        # Gracefully interrupt image generation
        print("KeyboardInterrupt: Exiting now.")
        exit(1)
