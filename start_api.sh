#!/bin/bash
# Start script for AI Proctoring FastAPI backend

echo "=================================="
echo "AI Proctoring System - FastAPI"
echo "=================================="
echo ""

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "uvicorn not found. Installing..."
    pip install uvicorn[standard]
fi

# Start the server
echo "Starting FastAPI server..."
echo "API URL: http://localhost:8000"
echo "Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Get the project root directory (where this script is located)
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run the server
uvicorn src.web.main:app --reload --host 0.0.0.0 --port 8000
