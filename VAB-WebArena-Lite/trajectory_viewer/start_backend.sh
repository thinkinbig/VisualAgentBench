#!/bin/bash

# Start the FastAPI backend server
echo "Starting Trajectory Tree Viewer Backend..."

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "Activating conda environment 'webarena'..."
    conda activate webarena
fi

# Change to backend directory
cd "$(dirname "$0")/backend"

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start the server
echo "Starting FastAPI server on http://localhost:8001"
python main.py
