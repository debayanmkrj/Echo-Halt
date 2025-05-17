import os
import uuid
import shutil
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our video analysis modules
from video_analyzer import VideoAnalyzer
from video_database import VideoDatabase

# Create FastAPI app
app = FastAPI(title="Video Similarity API")

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our video analyzer and database
video_db = VideoDatabase(index_path="./data/video_index", 
                         video_dir="./data/original_videos")
video_analyzer = VideoAnalyzer(model_name="resnet50")

# Create directories if they don't exist
os.makedirs("./data/temp", exist_ok=True)
os.makedirs("./data/original_videos", exist_ok=True)
os.makedirs("./data/video_index", exist_ok=True)

# API Models
class AnalysisResult(BaseModel):
    temp_path: str
    filename: str
    unique_percentage: float
    is_unique: bool
    similar_videos: List[Dict]

class UploadResult(BaseModel):
    success: bool
    message: str
    filename: str

@app.on_event("startup")
async def startup_event():
    """Initialize the video database index on startup"""
    await video_db.initialize()
    print("Video database initialized")

@app.get("/api/videos")
async def get_videos():
    """Get a list of all videos in the database"""
    videos = await video_db.get_all_videos()
    return videos

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_video(video: UploadFile = File(...)):
    """Analyze a video file and compare it to the database"""
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided")
    
    # Create a unique filename
    unique_filename = f"{uuid.uuid4()}-{video.filename}"
    temp_path = os.path.join("./data/temp", unique_filename)
    
    # Save uploaded file to temp directory
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")
    
    # Analyze video
    try:
        # Extract features from the uploaded video
        features = await video_analyzer.extract_features(temp_path)
        
        # Compare to existing videos
        similar_videos, scores = await video_db.find_similar_videos(features)
        
        # Calculate uniqueness as the inverse of the highest similarity score
        max_similarity = max(scores) if scores else 0
        unique_percentage = round(100 - max_similarity)
        is_unique = unique_percentage >= 70  # Our threshold for uniqueness
        
        # Format the similar videos list
        similar_videos_list = [
            {
                "filename": similar_videos[i],
                "overallSimilarity": round(scores[i]),
                "size": os.path.getsize(os.path.join("./data/original_videos", similar_videos[i]))
            }
            for i in range(len(similar_videos))
            if scores[i] > 30  # Only include videos with similarity > 30%
        ]
        
        # Sort by similarity (highest first)
        similar_videos_list.sort(key=lambda x: x["overallSimilarity"], reverse=True)
        
        return {
            "temp_path": temp_path,
            "filename": unique_filename,
            "unique_percentage": unique_percentage,
            "is_unique": is_unique,
            "similar_videos": similar_videos_list
        }
    
    except Exception as e:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/upload", response_model=UploadResult)
async def upload_video(temp_path: str = Form(...)):
    """Move a previously analyzed video to the permanent storage"""
    if not os.path.exists(temp_path):
        raise HTTPException(status_code=400, detail="Temporary file not found")
    
    try:
        # Get the filename from the temp path
        filename = os.path.basename(temp_path)
        
        # Move file to permanent storage
        final_path = os.path.join("./data/original_videos", filename)
        shutil.move(temp_path, final_path)
        
        # Extract features and add to the index
        features = await video_analyzer.extract_features(final_path)
        await video_db.add_video(filename, features)
        
        return {
            "success": True,
            "message": "Video uploaded successfully",
            "filename": filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)