#!/bin/bash
# ContextPilot Launch Script
# Double-click this file to start ContextPilot

cd "$(dirname "$0")"

echo "Starting ContextPilot..."
echo ""

# Check if backend is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Starting backend API server..."
    cd backend
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
    python -m uvicorn backend.mcp_server:api --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    cd ..
    sleep 2
    echo "Backend started (PID: $BACKEND_PID)"
else
    echo "Backend already running"
fi

# Start frontend
echo "Starting frontend..."
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Check if Tauri is available
if command -v cargo &> /dev/null && [ -d "src-tauri" ]; then
    echo "Launching desktop app..."
    npm run tauri:dev
else
    echo "Opening in browser..."
    npm run dev &
    sleep 3
    open http://localhost:3000
fi

echo ""
echo "ContextPilot is running!"
echo "Dashboard: http://localhost:3000"
echo "API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
