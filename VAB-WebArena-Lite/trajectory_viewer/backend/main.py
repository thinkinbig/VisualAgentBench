from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import json
import os
import glob
from pathlib import Path

app = FastAPI(title="Trajectory Tree Viewer API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directories
TRAJECTORY_DIR = Path("../../outputs/trajectory")
SCREENSHOTS_DIR = Path("../../outputs/screenshots")

@app.get("/")
async def root():
    return {"message": "Trajectory Tree Viewer API"}

@app.get("/trajectories")
async def get_trajectories():
    """Get list of available trajectory files"""
    try:
        trajectory_files = glob.glob(str(TRAJECTORY_DIR / "*.json"))
        trajectories = []
        
        for file_path in trajectory_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    trajectories.append({
                        "filename": os.path.basename(file_path),
                        "intent": data.get("intent", "Unknown"),
                        "run_id": data.get("run_id", "unknown"),
                        "created_at": os.path.getctime(file_path)
                    })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        # Sort by creation time (newest first)
        trajectories.sort(key=lambda x: x["created_at"], reverse=True)
        return trajectories
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trajectories/{filename}")
async def get_trajectory(filename: str):
    """Get specific trajectory data"""
    try:
        file_path = TRAJECTORY_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Trajectory not found")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/screenshots/{filename}")
async def get_screenshot(filename: str):
    """Get screenshot file"""
    try:
        file_path = SCREENSHOTS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Screenshot not found")
        
        return FileResponse(file_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)