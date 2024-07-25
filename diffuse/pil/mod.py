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

from PIL import ExifTags, Image, ImageOps


def calculate_gcd(a: int, b: int) -> int:
    """
    Calculate the greatest common divisor for the remainder of a ratio between two integers.
    """
    if b == 0:  # Base case for recursion
        return a  # Discovered the greatest common divisor
    return calculate_gcd(b, a % b)  # Recursive step


def calculate_aspect_ratio(image: Image) -> tuple[int, int]:
    """
    Calculate the image aspect ratio based on the dimensions from Image.size.
    """
    width, height = image.size
    gcd = calculate_gcd(width, height)
    return width // gcd, height // gcd


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


def resize_and_pad(image, target_width=3, target_height=2):
    # we need to scale up by the aspect ratio we need
    # then scale down by the aspect ratio we're deviating from
    # otherwise, it ends up being equivalent to the identity of the image ratio
    # e.g. (3 / 2) / (3 / 2) = (3 / 2) * (2 / 3) = (2 / 2) * (3 / 3) = 1 * 1 = 1
    width, height = image.size
    aspect_ratio_width, aspect_ratio_height = calculate_aspect_ratio(image)

    # this is fine with landscape, but may be problematic with portrait
    # if portrait has 16:9, then rotation results in 9:16,
    # note this is just an example and can't assume the given ratio
    # portrait would need to be padded to fit a 3:2 which ImageOps.pad can take care of
    # in portait, we would only need to scale the width. the only issue that we need to
    # scale the height proportionally. not sure how to go about that just yet.
    if aspect_ratio_width > target_width:
        new_height = int(width * target_width)
        padding = (new_height - height) // 2
        image = ImageOps.pad(image, border=(0, padding), fill=(255, 255, 255))
    elif current_aspect_ratio < target_aspect_ratio:
        new_width = int(height * target_aspect_ratio)
        padding = (new_width - width) // 2
        image = ImageOps.expand(image, border=(padding, 0), fill=(255, 255, 255))

    return image


def process_image(input_path, output_path, display):
    with Image.open(input_path) as img:
        img = correct_orientation(img)
        img = resize_and_pad(img, target_aspect_ratio=3 / 2)
        img.save(output_path)

        if display:
            img.show()


def process_image(input_path: str, output_path: str, display: bool) -> None:
    with Image.open(input_path) as image:
        image = correct_orientation(image)
        width, height = image.size

        # Pad width to ensure it is divisible by 3
        aspect_ratio = calculate_aspect_ratio(image)  # -> tuple[int, int]
        if aspect_ratio != (3, 2):
            # we need to scale up by the aspect ratio we need
            # then scale down by the aspect ratio we're deviating from
            # otherwise, it ends up being equivalent to the identity of the image ratio
            # e.g. (3 / 2) / (3 / 2) = (3 / 2) * (2 / 3) = (2 / 2) * (3 / 3) = 1 * 1 = 1
            ar_width = aspect_ratio[0]
            ar_height = aspect_ratio[1]

            # this is fine with landscape, but may be problematic with portrait
            # if portrait has 16:9, then rotation results in 9:16,
            # note this is just an example and can't assume the given ratio
            # portrait would need to be padded to fit a 3:2 which ImageOps.pad can take care of
            # in portait, we would only need to scale the width. the only issue that we need to
            # scale the height proportionally. not sure how to go about that just yet.
            new_width = width * 3  # scale the width by 3
            new_height = height * 2  # scale the height by 2

            padding = new_width - width
            # Returns a resized and padded version of the image,
            # expanded to fill the requested aspect ratio and size.
            image = ImageOps.pad(
                image=image,
                size=(padding // 2, 0),
                method=Image.Resampling.BICUBIC,
                color=(255, 255, 255),
                centering=(0.5, 0.5),
            )

        # Scale down until an aspect ratio of 3:2 is achieved
        # NOTE: ANTIALIAS was superseded by LANCZOS starting from Pillow 2.7.0
        image = image.resize((new_width, int(new_width / 3 * 2)), Image.LANCZOS)

        image.save(output_path)

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
