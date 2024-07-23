# diffuse

Simple experimental CLI scripts for stable diffusion models.

## About

Welcome to the `diffuse` repository, where I have gathered utility scripts developed in my spare time for experimenting with Stable Diffusion models. This project is simply a experimental repostiory that acts a central hub CLI scripts.

## Supported Models

Currently supported models include stable-diffusion-xl-1.0, stable-diffusion-xl-turbo, and stable-diffusion-3-medium.

## Installation

Note that PyTorch will default to installing the CPU-only version for users lacking CUDA support.

To get started with the `diffuse` project, follow these steps:

1. Create and activate the virtual environment:

```sh
virtualenv .venv
source .venv/bin/activate
```

1. Enable the execution bit for setup scripts:

```sh
chmod +x torch.sh
chmod +x setup.sh
```

1. Run the provided setup script to install dependencies and set up your environment:

```sh
./setup.sh
```

## Contributing

Contributions to `diffuse` are welcome! If you have any improvements, bug fixes, or new features in mind for this project, please open a pull request and let's discuss the best approach together.

Happy coding! 🤖🚀
