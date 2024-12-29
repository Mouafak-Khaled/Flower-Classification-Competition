#!/bin/bash

set -e  # Exit script immediately on any error

ENV_NAME="flower_cls_env"
PYTHON_VERSION="3.12"
REQ_FILE="requirements.txt"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

# Check if the environment already exists
if ! conda info --envs | grep -q "^$ENV_NAME\s"; then
    # Create new environment
    echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create --name $ENV_NAME python=$PYTHON_VERSION
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

# Activate the environment
echo "Activating Conda environment '$ENV_NAME'..."
source activate $ENV_NAME || { echo "Failed to activate Conda environment '$ENV_NAME'"; exit 1; }

# Install PyTorch with specific index URL
echo "Installing PyTorch, Torchvision, and Torchaudio from '$TORCH_INDEX_URL'..."
python -m pip install torch torchvision torchaudio --index-url $TORCH_INDEX_URL


# Install required dependencies
echo "Installing Python packages from '$REQ_FILE'..."
python -m pip install -r $REQ_FILE

echo "Setup completed successfully."
