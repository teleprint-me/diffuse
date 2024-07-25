#!/usr/bin/env bash

# install torch for cpu by default
pip install torch torchvision torchtext --index-url https://download.pytorch.org/whl/cpu --upgrade
# install required dependencies
pip install -r requirements.txt --upgrade
