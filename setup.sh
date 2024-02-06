#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "Virtual environment activated."

# Install libraries
pip install -r requirements.txt

# Deactivate virtual environment
#echo "Deactivating virtual environment..."
#deactivate
#echo "Virtual environment deactivated"
