#!/bin/bash

# Start the React frontend development server
echo "Starting Trajectory Tree Viewer Frontend..."

# Change to frontend directory
cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

# Start the development server
echo "Starting React development server on http://localhost:3000"
npm start
