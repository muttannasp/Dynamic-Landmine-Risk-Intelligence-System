#!/bin/bash

# Start script for Dynamic Landmine Risk Intelligence System (Working Version)
echo "🚀 Starting Dynamic Landmine Risk Intelligence System (React-Style Interface)..."

# Function to kill background processes on exit
cleanup() {
    echo "🛑 Stopping services..."
    kill $STREAMLIT_PID 2>/dev/null
    exit
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start Streamlit app
echo "🌐 Starting Streamlit app on port 8502..."
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run app_react_style.py --server.port 8502 &
STREAMLIT_PID=$!

echo "✅ Service started successfully!"
echo "🌐 App: http://localhost:8502"
echo ""
echo "🎯 Features:"
echo "   • Interactive map with clickable markers"
echo "   • Feature explanation tool (select points from dropdown)"
echo "   • Real-time model performance metrics"
echo "   • Add new data points and retrain model"
echo ""
echo "Press Ctrl+C to stop the service"

# Wait for process
wait
