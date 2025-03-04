#!/bin/bash

set -e  # Exit on error

ENV_NAME="flower_cls_env"

# Check if the environment exists
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "Activating existing Conda environment '$ENV_NAME'..."
    source activate $ENV_NAME || { echo "Failed to activate Conda environment '$ENV_NAME'"; exit 1; }
else
    echo "Conda environment '$ENV_NAME' not found. Running setup.sh..."
    bash setup.sh
fi
