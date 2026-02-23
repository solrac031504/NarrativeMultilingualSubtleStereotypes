#!/bin/bash

# Exit if error
set -e

ENV_NAME="NarrativeMultilingualSubtleStereotypes"
REQUIREMENTS_FILE="requirements.txt"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv "$ENV_NAME"

# Active virtual environment
echo "Activating virtual environment..."
source "$ENV_NAME/bin/activate"

# Install dependencies with pip
echo "Installing dependencies from $REQUIREMENTS_FILE..."
pip install -r "$REQUIREMENTS_FILE"

echo "Installation complete. Virtual environment is active."