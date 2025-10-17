#!/bin/bash

# Start script for Dynamic Landmine Risk Intelligence System
echo "🚀 Starting Dynamic Landmine Risk Intelligence System..."

# Function to kill background processes on exit
cleanup() {
    echo "🛑 Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start backend
echo "📡 Starting FastAPI backend on port 8000..."
cd "$(dirname "$0")"
source venv/bin/activate
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "🌐 Starting React frontend on port 3001..."
cd frontend
npm start &
FRONTEND_PID=$!

echo "✅ Services started successfully!"
echo "📡 Backend API: http://localhost:8000"
echo "🌐 Frontend: http://localhost:3001"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for processes
wait
