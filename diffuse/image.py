"""
Module: diffuse.image

References:
- PIL Image API
    - https://pillow.readthedocs.io/en/stable/reference/Image.html
- PIL Exif API
    - https://pillow.readthedocs.io/en/stable/reference/ExifTags.html
- Calculating aspect ratios
    - https://stackoverflow.com/questions/1186414/whats-the-algorithm-to-calculate-aspect-ratio
- Calculating image dimensions and transpose
    - https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image
- ANTIALIAS renamed to LANCZOS
    - https://pillow.readthedocs.io/en/stable/releasenotes/2.7.0.html#antialias-renamed-to-lanczos
"""

import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from PIL import ExifTags, Image


def evaluate_path(path: str) -> str:
    """
    Evaluate and expand the given file path
    """
    return os.path.expanduser(os.path.expandvars(path))


def float_is_close(
    a: float,
    b: float,
    relative: float = 1e-03,
    absolute: float = 0.0,
) -> bool:
    """
    Check if two floating-point values are approximately equal within specified tolerances.
    """
    return abs(a - b) <= max(relative * max(abs(a), abs(b)), absolute)


def calculate_gcd(a: int, b: int) -> int:
    """
    Calculate the greatest common divisor for the remainder of a ratio between two integers.
    """
    if b == 0:  # Base case for recursion
        return a  # Discovered the greatest common divisor
    return calculate_gcd(b, a % b)  # Recursive step


def calculate_aspect_ratio(image: Image) -> Tuple[int, int]:
    """
    Calculate the image aspect ratio based on the dimensions from Image.size.
    """
    width, height = image.size
    gcd = calculate_gcd(width, height)
    return width // gcd, height // gcd


def calculate_dimensions(
    image: Image, aspect_ratios: Optional[List[Tuple[int, int]]] = None
) -> Tuple[int, int]:
    """
    Calculate new dimensions based on the closest target aspect ratio.
    """
    width, height = image.size

    if aspect_ratios is None:
        aspect_ratios = [(4, 3), (3, 2)]

    current_ar = width / height
    closest_ar = min(aspect_ratios, key=lambda ar: abs(current_ar - ar[0] / ar[1]))
    target_width, target_height = closest_ar

    if width / target_width > height / target_height:
        new_width = width
        new_height = int(width * target_height / target_width)
    else:
        new_height = height
        new_width = int(height * target_width / target_height)

    return new_width, new_height


def correct_orientation(image: Image) -> Image:
    """
    Correct the image orientation based on EXIF data.
    """
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass  # No EXIF data, or unexpected format
    return image


def resize_and_pad(image: Image, new_width: int, new_height: int) -> Image:
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    max_dim = max(new_width, new_height)
    new_image = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    box_width = (max_dim - new_width) // 2
    box_height = (max_dim - new_height) // 2
    new_image.paste(resized_image, (box_width, box_height))
    return new_image


def initialize_image(
    image_path: str,
    dimensions: Optional[Tuple[int, int]] = None,
    aspect_ratios: Optional[List[Tuple[int, int]]] = None,
) -> Image:
    image = Image.open(image_path).convert("RGB")
    image = correct_orientation(image)

    if dimensions is None:
        new_width, new_height = calculate_dimensions(image, aspect_ratios)
    else:
        new_width, new_height = dimensions

    return resize_and_pad(image, new_width, new_height)


def get_estimated_steps(
    strength: float,
    num_inference_steps: int,
) -> int:
    """
    Estimate the number of inference steps based on the strength parameter.

    This function simulates the calculation performed internally by the StableDiffusionXLImg2ImgPipeline in the
    diffusers library when processing image-to-image tasks. It determines the effective number of inference
    steps that will be used by the pipeline, considering the influence of the strength parameter.

    Parameters:
    - strength (float): The amount of noise to add to the image, typically between 0 and 1.
    - num_inference_steps (int): The initially desired number of inference steps.

    Returns:
    - int: The estimated number of inference steps after considering the strength parameter.

    Note:
    - In the context of the StableDiffusionXLImg2ImgPipeline, the strength parameter affects the starting point of the
      denoising process. A lower strength value reduces the number of effective inference steps. This function mimics
      that behavior by scaling down the number of steps based on the strength value.

    Example:
    >>> get_estimated_steps(0.5, 50)
    25
    >>> get_estimated_steps(1.0, 50)
    50
    """
    return min(int(num_inference_steps * strength), num_inference_steps)


def adjust_inference_steps(
    strength: float,
    num_inference_steps: int,
    allow_low_strength: bool = False,
) -> int:
    """
    Adjust the number of inference steps for the StableDiffusionXLImg2ImgPipeline based on the strength parameter.

    The function scales the number of inference steps to compensate for the reduction caused by lower strength values.
    It ensures that the effective number of inference steps remains close to the desired amount, despite the strength parameter's influence.

    Parameters:
    - strength (float): The amount of noise to add to the image, typically between 0 and 1.
    - num_inference_steps (int): The desired number of inference steps.
    - allow_low_strength (bool): If set to True, allows strength values below 10%. Defaults to False.

    Returns:
    - int: The adjusted number of inference steps.

    Raises:
    - ValueError: If strength is below 10% and allow_low_strength is False. Lower strength values significantly alter the number of inference steps, leading to unpredictable results. This behavior can be bypassed by specifying a denoising start value.

    Note:
    - This function normalizes the strength value to a percentage for easier comparison and decision-making.
    - It's tailored to the behavior of the StableDiffusionXLImg2ImgPipeline in the diffusers library, where the strength parameter influences the starting point of the denoising process.

    Example:
    >>> adjust_inference_steps(0.5, 50)
    50
    >>> adjust_inference_steps(0.05, 50)
    ValueError: Strength must be at least 10% unless a denoising start value is specified.
    """
    estimated_strength = strength * 100  # normalize the float value
    if estimated_strength < 10 and not allow_low_strength:
        raise ValueError(
            "Strength must be at least 10% unless a denoising start value is specified."
        )

    # Calculate the original estimated inference steps
    original_estimated_steps = get_estimated_steps(strength, num_inference_steps)

    # Determine the adjustment factor
    adjustment_factor = num_inference_steps / max(original_estimated_steps, 1)

    # Apply the adjustment
    return int(num_inference_steps * adjustment_factor)


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
    output_directory: str = "images",
    delay: float = 1 / 30,
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
        time.sleep(delay)  # NOTE: Prevent overwrites

    return dataset


def generate_image_to_image(
    pipe_image: DiffusionPipeline,
    image_path: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    strength: float = 0.75,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    num_images_per_prompt: int = 2,
    output_directory: str = "images",
    delay: float = 1 / 30,
    dimensions: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Tuple[Image, str]], float]:
    try:
        start_time = datetime.now()

        os.makedirs(output_directory, exist_ok=True)

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
