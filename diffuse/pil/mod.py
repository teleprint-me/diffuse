#!/usr/bin/env python

"""
diffuse.pil.mod

A script for preprocessing images to meet the requirements of diffusion models, which expect an aspect ratio of 3:2.

The script handles two main cases:

1. Landscape images with a valid aspect ratio are left unchanged.
2. Portrait images (with a different aspect ratio) undergo three transformations before being adjusted to the required dimensions:
   - Rotate back to their original orientation using metadata if available
   - Pad the width to ensure it is divisible by 3
   - Scale down until an aspect ratio of 3:2 is achieved

This script aims to simplify the preprocessing step for stable diffusion models, making it easier to implement and use various images as input. In most cases, this script isn't necessary, but it can help save time when dealing with a large number of images or complex image formats.
"""

import argparse
import os
from datetime import datetime

from PIL import ExifTags, Image, ImageOps


def correct_orientation(image: Image) -> Image:
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


def resize_and_pad(image: Image, target_aspect_ratio: float) -> Image:
    width, height = image.size
    aspect_ratio = width / height

    if aspect_ratio > target_aspect_ratio:
        new_height = int((width / target_aspect_ratio))
        image = ImageOps.pad(image, (width, new_height), color=(0, 0, 0))
    elif aspect_ratio < target_aspect_ratio:
        new_width = int((height * target_aspect_ratio))
        image = ImageOps.pad(image, (new_width, height), color=(0, 0, 0))

    return image.resize((width, int(width / target_aspect_ratio)), Image.LANCZOS)


def write_image(image: Image, output_path: str) -> None:
    image_path = f"{output_path}/{datetime.now()}.png"
    image.save(image_path)
    print(f"Saved image to {image_path}")


def process_image(input_path: str, output_path: str, display: bool) -> None:
    with Image.open(input_path) as image:
        # orientate, resize, and pad image
        image = correct_orientation(image)
        image = resize_and_pad(image, target_aspect_ratio=3 / 2)
        image = image.resize((1200, 800), Image.LANCZOS)  # Resize to fixed size

        # create the path, file, and write to disk
        os.makedirs(output_path, exist_ok=True)
        write_image(image, output_path)

        if display:
            image.show()


def get_parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess images for diffusion models with aspect ratio 3:2."
    )
    parser.add_argument("input", type=str, help="Input image file path")
    parser.add_argument(
        "--output",
        type=str,
        default="images",
        help="Output directory path. Default is 'images'.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the resulting image. Default is False.",
    )
    return parser.parse_args()


def main():
    args = get_parser_arguments()
    process_image(args.input, args.output, args.display)


if __name__ == "__main__":
    main()
