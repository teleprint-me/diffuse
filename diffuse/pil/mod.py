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

from PIL import Image

from diffuse.image import image_initialize


def write_image(image: Image, output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)
    image_path = f"{output_path}/{datetime.now()}.png"
    image.save(image_path)
    print(f"Saved image to {image_path}")


def process_image(
    input_path: str,
    output_path: str,
    display: bool,
    dimensions: tuple,
) -> None:
    with Image.open(input_path) as image:
        # orientate, resize, and pad image
        image = image_initialize(input_path, dimensions)

        # create the path, file, and write to disk
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
        "--size",
        type=str,
        choices=["small", "medium", "large"],
        default="small",
        help="The size of the output image. Default is 'small'.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the resulting image. Default is False.",
    )
    return parser.parse_args()


def main():
    args = get_parser_arguments()

    dimensions = {
        "small": (900, 600),
        "medium": (1200, 800),
        "large": (1500, 1000),
    }.get(args.size)

    process_image(args.input, args.output, args.display, dimensions)


if __name__ == "__main__":
    main()
