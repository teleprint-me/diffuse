"""
Module: diffuse.image
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
    return os.path.expanduser(os.path.expandvars(path))


def float_is_close(
    a: float,
    b: float,
    relative: float = 1e-03,
    absolute: float = 0.0,
):
    """
    Check if two floating-point values are approximately equal within specified tolerances.
    """
    return abs(a - b) <= max(relative * max(abs(a), abs(b)), absolute)


# sauce: https://stackoverflow.com/questions/1186414/whats-the-algorithm-to-calculate-aspect-ratio
def gcd(a: int, b: int) -> int:
    if b == 0:  # hit bedrock and is no longer divisible.
        return a  # discovered the greatest common divisor
    return gcd(b, a % b)  # keep factoring


def aspect_ratio(width: int, height: int) -> tuple[int, int]:
    divisor = gcd(width, height)
    return width // divisor, height // divisor


def calculate_dimensions(
    width: int,
    height: int,
    aspect_ratios: Optional[list[tuple[int, int]]] = None,
) -> tuple[int, int]:
    # Desired aspect ratios
    if aspect_ratios is None:
        aspect_ratios = [(4, 3), (3, 2)]

    # Calculate current aspect ratio
    current_ar = width / height

    # Find closest aspect ratio
    closest_ar = min(aspect_ratios, key=lambda ar: abs(current_ar - ar[0] / ar[1]))

    # Calculate new dimensions based on the closest aspect ratio
    new_width = closest_ar[0] * height
    new_height = closest_ar[1] * width

    if new_width > width:
        new_width = width
        new_height = int(width * closest_ar[1] / closest_ar[0])
    else:
        new_height = height
        new_width = int(height * closest_ar[0] / closest_ar[1])

    return new_width, new_height


# sauce: https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image
def correct_orientation(image: Image) -> Image:
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
        # guess if image does not have getexif
        pass
    return image


def initialize_image(
    image_path: str,
    dimensions=None,
    aspect_ratio=(3, 2),
) -> Image:
    image = Image.open(image_path).convert("RGB")

    # Correct image orientation based on EXIF data
    image = correct_orientation(image)

    # Handle portrait and landscape mode by ensuring width and height are assigned correctly
    height, width = image.size

    if dimensions is None:
        new_width, new_height = calculate_dimensions(width, height)
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

        # Create new image with padding to fit the aspect ratio exactly
        new_image = Image.new(
            "RGB", (max(new_width, new_height), max(new_width, new_height)), (0, 0, 0)
        )
        new_image.paste(
            resized_image,
            ((new_image.width - new_width) // 2, (new_image.height - new_height) // 2),
        )
    else:
        resized_image = image.resize(dimensions, Image.ANTIALIAS)
        new_image = resized_image

    return new_image


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
