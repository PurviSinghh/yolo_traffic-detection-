import uvicorn
import sys
import os

if __name__ == "__main__":
    # Add root directory to python path if needed
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("🚦 Starting Smart Traffic Management Dashboard...")
    print("🌐 The web interface will be available at: http://localhost:8000")
    
    # Run the FastAPI server
    # server.py imports its siblings as top-level modules, so serve from backend/
    uvicorn.run("server:app", app_dir="backend", host="0.0.0.0", port=8000, reload=True)
