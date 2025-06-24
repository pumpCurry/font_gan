#!/usr/bin/env bash
# setup.sh - Create virtual environment and install packages on Linux/WSL

python3 -m venv font_gan_venv
source font_gan_venv/bin/activate
pip install --upgrade pip
# Install PyTorch for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Additional libraries
pip install Pillow scipy opencv-python scikit-image tensorboard
# Optional packages
pip install optuna fontforge

echo "Environment setup complete."

