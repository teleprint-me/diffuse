"""
Script: diffuse.pil.mod

Script to modify an image if needed.

Diffusion models expect images with an aspect ratio of 3:2

In most cases, landscape images are fine, but may need to be rescaled to fit a ratio of 3:2.

Portrait images cause issues as they are rotated 90 degrees in most cases. The image needs to be rotated back to its expected orientation with the use of metadata. Then the image needs a padded width. Finally, the image needs to be scaled to a aspect ratio of 3:2.

This script attempts to completely automate this process so that any image may be used as input for any compatible stable diffusion model.

In most cases, this script shouldn't be necessary, but it's being implemented to assist with implementing the boilerplate code necessary to achieve the related end goals.
"""
