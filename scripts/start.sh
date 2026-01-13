#!/bin/bash

# ConvoBench Startup Script
# Starts both the API server and frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting ConvoBench..."
echo ""

# Check for .env file
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env" 2>/dev/null || true
    echo "   Please add your NVIDIA_API_KEY to .env"
    echo ""
fi

# Start API server
echo "📡 Starting API server on http://localhost:8000"
cd "$PROJECT_DIR"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait for API to be ready
sleep 2

# Start frontend
echo "🌐 Starting frontend on http://localhost:3000"
cd "$PROJECT_DIR/frontend"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "   Installing frontend dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ ConvoBench is running!"
echo "   Frontend: http://localhost:3000"
echo "   API:      http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all services"

# Handle shutdown
trap "kill $API_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM

# Wait for processes
wait
