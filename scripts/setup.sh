#!/bin/bash

# ConvoBench Setup Script
# Creates a virtual environment and installs dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🔧 Setting up ConvoBench..."
echo ""

cd "$PROJECT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -e .

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  No .env file found. Creating from template..."
    cp .env.example .env 2>/dev/null || echo "NVIDIA_API_KEY=your-key-here" > .env
    echo "   Please add your NVIDIA_API_KEY to .env"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start ConvoBench:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Start the API server:"
echo "     uvicorn api.main:app --reload --port 8000"
echo ""
echo "  3. In another terminal, start the frontend:"
echo "     cd frontend && npm install && npm run dev"
echo ""
echo "  4. Open http://localhost:3000"
