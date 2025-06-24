@echo off
REM setup_windows.bat - Create virtual environment and install requirements

python -m venv font_gan_venv
call font_gan_venv\Scripts\activate
python -m pip install --upgrade pip
REM Install PyTorch (CUDA 11.8 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
REM Additional libraries
pip install Pillow scipy opencv-python scikit-image tensorboard
REM Optional packages
pip install optuna fontforge

echo Environment setup complete.

