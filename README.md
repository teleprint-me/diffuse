# diffuse

Simple experimental CLI scripts for stable diffusion models.

## About

Welcome to `diffuse`, an experimental repository housing utility scripts I've developed in my spare time as part of my journey exploring Stable Diffusion Models (SDMs). This project serves as a central hub for these CLI-focused scripts, allowing easy execution and learning about the diffusers API.

## Supported Models

Currently supported models include `stable-diffusion-xl-1.0`, `stable-diffusion-xl-turbo`, and `stable-diffusion-3-medium`.

## Installation

Note that PyTorch will default to installing the CPU-only version for users lacking CUDA support.

To get started with `diffuse`:

1. Create and activate a virtual environment:

```sh
virtualenv .venv
source .venv/bin/activate
```

2. Enable execution bit for setup scripts:

```sh
chmod +x setup.sh
```

3. Run the provided `setup.sh` script to install dependencies and set up your environment:

```sh
./setup.sh
```

## Contributing

Contributions are welcome! If you have ideas for improvements, bug fixes, or new features in mind for this project, please open a pull request, and we can discuss the best approach together. Happy coding! 🤖🚀
