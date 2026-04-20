#!/bin/bash

# Exit immediately if any command fails
set -e

echo "=========================================="
echo "🚀 Demeter: Automating Kaggle Push"
echo "=========================================="

# Ensure we are in the right directory before doing anything destructive
if [ ! -d "kaggle_upload" ] || [ ! -d "src" ]; then
    echo "❌ Error: Please run this script from the root 'ENGG2112_Demeter' directory."
    exit 1
fi

# Step 1: Overwrite staging area with the latest Python modules
echo "[1/3] Staging Python modules from src/..."
cp -v src/*.py kaggle_upload/

# Step 2: Ensure the manual data folders are synced (ignoring errors if they are empty)
echo "[2/3] Staging manual dataset folders..."
cp -r data/layer2_manual_thermal kaggle_upload/ 2>/dev/null || true
cp -r data/layer3_manual_danforth kaggle_upload/ 2>/dev/null || true

# Step 3: Trigger the Kaggle CLI
echo "[3/3] Pushing new version to Kaggle..."
cd kaggle_upload

# Generate a dynamic timestamp so you know exactly which version is which on Kaggle
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Push the update. The '-m' flag adds your version notes.
kaggle datasets version -m "Automated Demeter pipeline update: $TIMESTAMP"

echo "=========================================="
echo "✅ Push Complete! Your Kaggle notebook can now access the latest code."
echo "=========================================="
