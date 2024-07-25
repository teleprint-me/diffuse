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
