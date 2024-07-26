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

import time
from datetime import datetime
from typing import List, Optional, Tuple

from PIL import ExifTags, Image, ImageOps


def image_correct_orientation(image: Image) -> Image:
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif[orientation]
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image


def image_pad_and_resize(image: Image, target_aspect_ratio: float) -> Image:
    width, height = image.size
    aspect_ratio = width / height

    if aspect_ratio > target_aspect_ratio:
        new_height = int((width / target_aspect_ratio))
        image = ImageOps.pad(image, (width, new_height), color=(0, 0, 0))
    elif aspect_ratio < target_aspect_ratio:
        new_width = int((height * target_aspect_ratio))
        image = ImageOps.pad(image, (new_width, height), color=(0, 0, 0))

    return image.resize((width, int(width / target_aspect_ratio)), Image.LANCZOS)


def image_initialize(
    image_path: str,
    dimensions: Optional[tuple[int, int]] = None,
) -> Image.Image:
    with Image.open(image_path) as image:
        # orientate, resize, and pad image
        image = image_correct_orientation(image)
        image = image_pad_and_resize(image, target_aspect_ratio=3 / 2)

        # Resize to fixed size
        if dimensions is None:
            image = image.resize((1200, 800), Image.LANCZOS)
        else:
            image = image.resize(dimensions, Image.LANCZOS)

    return image


def image_write(
    images: List[Image],
    output_directory: Optional[str] = None,
    delay: Optional[float] = None,
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

    if output_directory is None:
        output_directory = "images"

    if delay is None:
        delay = 1 / 30

    for image in images:
        image_path = f"{output_directory}/{datetime.now()}.png"
        image.save(image_path)
        dataset.append((image, image_path))
        print(f"Created: {image_path}")
        time.sleep(delay)  # NOTE: Prevent overwrites

    return dataset
