"""
Module: diffuse.text
"""

from datetime import datetime
from time import sleep

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
