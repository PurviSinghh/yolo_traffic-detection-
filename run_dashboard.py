import uvicorn
import sys
import os

if __name__ == "__main__":
    # Add root directory to python path if needed
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("🚦 Starting Smart Traffic Management Dashboard...")
    print("🌐 The web interface will be available at: http://localhost:8000")
    
    # Run the FastAPI server
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
