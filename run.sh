#!/bin/zsh

rm -rf .venv
# Create a new virtual environment
python3.10 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install PyTorch with MPS support
pip3.10 install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# Install other requirements
pip3.10 install -r requirements.txt


# python3.10 src/test_setup.py
# python3.10 src/main.py
