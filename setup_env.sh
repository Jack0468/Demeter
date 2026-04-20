#!/bin/bash

# Exit the script immediately if any command fails
set -e

# Define environment variables
ENV_NAME="demeter_env"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "🌱 Starting Demeter AI Environment Setup"
echo "=========================================="

# 1. Create the Conda environment (The '-y' flag automatically says 'yes' to prompts)
echo "[1/4] Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 2. Hook Conda into the bash script so we can use 'conda activate'
echo "[2/4] Hooking Conda into script..."
eval "$(conda shell.bash hook)"

echo "Activating '$ENV_NAME'..."
conda activate $ENV_NAME

# 3. Install CMake first to prevent C++ compilation errors (like the dm-tree bug)
echo "[3/4] Installing CMake and core build tools..."
conda install cmake -y

# 4. Install all the required Machine Learning Python libraries
echo "[4/4] Installing Python dependencies via pip..."
pip install --no-cache-dir tensorflow pandas numpy scikit-learn kaggle joblib tensorflow-datasets Pillow

echo "=========================================="
echo "✅ Environment Setup Complete!"
echo "=========================================="
echo "To begin working on Demeter, run this command in your terminal:"
echo "conda activate $ENV_NAME"
echo "=========================================="