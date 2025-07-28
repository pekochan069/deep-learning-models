# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning models repository implementing various neural network architectures using PyTorch. The project uses a configuration-driven approach with JSON configs to define training parameters, model selection, and dataset choices.

## Commands

### Development Commands

- **Run main training**: `python src/main.py` (or `uv run poe main`)
- **Format code**: `uv run poe format` (uses `ruff`)
- **Lint code**: `uv run poe lint` (uses `ruff check --fix`)

### Package Management

- Uses `uv` for dependency management
- Install dependencies: `uv sync`
- Install dev dependencies: `uv sync --group dev`

## Architecture

### Core Components

**Configuration System** (`src/core/config.py`):

- Uses Pydantic models for type-safe configuration
- JSON configs stored in `configs/` directory
- Config includes model type, dataset, optimizer, loss function, training parameters
- Supports early stopping, learning rate scheduling

**Base Model** (`src/models/base_model.py`):

- Abstract base class that all models inherit from
- Implements common training loop, validation, early stopping
- Handles model saving/loading with safetensors format
- Provides training history plotting and model summary

**Model Factory** (`src/models/__init__.py`):

- Central factory function `get_model(config)` to instantiate models
- Supports 20+ model architectures including ResNet, DenseNet, EfficientNet, Inception, VGG, etc.

### Directory Structure

- `src/core/`: Core utilities (config, dataset, device, logger, loss, optimizer, weights)
- `src/models/`: Model implementations organized by architecture family
- `configs/`: JSON configuration files for different training runs
- `weights/`: Saved model weights in safetensors format
- `images/`: Training loss plots
- `data/`: Datasets (MNIST, CIFAR-10, ImageNet)

### Model Implementation Pattern

All models inherit from `BaseModel` and implement:

- `train_epoch()`: Single training epoch implementation
- `validate_epoch()`: Validation logic
- `predict()`: Inference on test data

### Configuration-Driven Training

1. Create/modify JSON config in `configs/` with model, dataset, optimizer settings
2. Update `src/main.py` to load the desired config
3. Run training with `python src/main.py`
4. Models auto-save weights and generate loss plots

### Supported Models

- CNN: LeNet, AlexNet, VGG (11/13/16/19), ResNet (18/34/50/101/152)
- Modern: Inception (v1/v2), DenseNet (121/169/201/264), MobileNet, ShuffleNet
- Efficient: EfficientNet (B0-B8, L2)
- Generative: GAN

### Datasets

- MNIST, CIFAR-10, ImageNet, Mini-ImageNet
- Custom dataset loader in `src/core/dataset.py`
