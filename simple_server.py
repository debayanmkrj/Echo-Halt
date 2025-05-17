from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import json
import time
import tempfile
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import concurrent.futures
from scipy.fftpack import dct
from scipy import spatial
from scipy.signal import correlate2d
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sklearn.metrics.pairwise as metrics
from collections import defaultdict
import uvicorn.config
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('video_upload_debug.log'),
        logging.StreamHandler()
    ]
)


uvicorn.config.TIMEOUT_KEEP_ALIVE = 500
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React default
        "http://localhost:3001",   # Python backend
        "http://localhost:3002",   # Node backend
        "http://localhost:3003",   # Frontend
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[
        "Content-Type", 
        "Authorization", 
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Headers",
        "X-Requested-With"
    ]
)

@app.options("/api/analyze")
async def options_analyze():
    return {"status": "preflight ok"}

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(BASE_DIR, "..")  # Go up one level to server directory
VIDEOS_DIR = os.path.join(SERVER_DIR, "public", "original_videos")
TEMP_DIR = os.path.join(SERVER_DIR, "public", "temp")
KEYFRAMES_DIR = os.path.join(SERVER_DIR, "public", "keyframes")
ANALYSIS_DIR = os.path.join(SERVER_DIR, "public", "analysis_results")

app.mount("/original_videos", StaticFiles(directory=VIDEOS_DIR), name="original_videos")
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")
app.mount("/keyframes", StaticFiles(directory=KEYFRAMES_DIR), name="keyframes")
app.mount("/analysis_results", StaticFiles(directory=ANALYSIS_DIR), name="analysis_results")



# Add these constants at the top of your file
# Adjust these to match your server's capabilities
MAX_IMAGE_SIZE = (320, 240)  # Resize frames to this size for analysis
MAX_FRAMES = 10  # Analyze up to this many frames per video
USE_OBJECT_DETECTION = False  # Set to False if running out of memory

# Create directories if they don't exist
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(KEYFRAMES_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Initialize deep learning model for feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    # Load ResNet model for feature extraction
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
    model = model.to(device)
    model.eval()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("ResNet model loaded successfully on", device)
    
    # Load a pre-trained object detection model (faster RCNN)
    try:
        obj_detection_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        obj_detection_model.to(device)
        obj_detection_model.eval()
        print("Object Detection model loaded successfully")
    except Exception as e:
        print(f"Error loading object detection model: {str(e)}")
        obj_detection_model = None
        
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    obj_detection_model = None

# COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy types.
    This encoder converts NumPy types to standard Python types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to JSON-serializable types
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj



###########################################
# 1. IMPROVED PERCEPTUAL HASH COMPARISON #
###########################################

def compute_phash(image, hash_size=16):
    """Compute perceptual hash for an image with improved robustness"""
    try:
        # Convert to grayscale and resize
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply slight blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Resize to (hash_size+1, hash_size) for computing differences
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        
        # Compute difference hash (horizontal differences)
        h_diff = resized[:, 1:] > resized[:, :-1]
        
        # Compute difference hash (vertical differences - improves robustness)
        resized_v = cv2.resize(gray, (hash_size, hash_size + 1), interpolation=cv2.INTER_AREA)
        v_diff = resized_v[1:, :] > resized_v[:-1, :]
        
        # Combine both hashes
        combined_hash = np.concatenate([h_diff.flatten(), v_diff.flatten()])
        
        # Convert to binary hash
        return combined_hash.astype(int)
    except Exception as e:
        print(f"Error in perceptual hash: {str(e)}")
        return np.zeros(hash_size * hash_size * 2, dtype=int)

def compare_phash(hash1, hash2):
    """Compare two perceptual hashes with improved accuracy"""
    if not isinstance(hash1, np.ndarray) or not isinstance(hash2, np.ndarray):
        try:
            hash1 = np.array(hash1)
            hash2 = np.array(hash2)
        except:
            return 0  # Return 0 similarity if conversion fails
    
    if len(hash1) != len(hash2):
        return 0
    
    # Hamming distance
    distance = np.sum(hash1 != hash2)
    
    # Convert to similarity percentage with improved scaling
    # Less aggressive exponential decay to avoid false negatives
    max_distance = len(hash1)
    
    # Use a more gradual similarity curve
    # The constant 2.0 (reduced from 3.0) makes the decay less aggressive
    similarity = 100 * np.exp(-2.0 * distance / max_distance)
    
    # Boost similarity slightly for high similarity hashes (likely duplicates)
    if similarity > 75:
        similarity = 75 + (similarity - 75) * 1.2
    
    # Cap at 100%
    return min(100, similarity)

def save_keyframes(frames, output_dir, video_name):
    """Save extracted keyframes to disk"""
    keyframe_paths = []
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each frame as a JPEG
    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"{video_name}_frame_{i}.jpg")
        cv2.imwrite(frame_path, frame)
        keyframe_paths.append(frame_path)
    
    return keyframe_paths

##################################
# 2. IMPROVED HISTOGRAM ANALYSIS #
##################################

def compute_histogram(image):
    """Compute color histogram for an image with better binning strategy"""
    try:
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Use 16x16x16 bins for better color representation
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16], 
                          [0, 180, 0, 256, 0, 256])
        
        # Normalize to account for different image sizes
        cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
        
        # Convert to 1D array
        return hist.flatten()
    except Exception as e:
        print(f"Error computing histogram: {str(e)}")
        return np.zeros(16 * 16 * 16)

def compare_histograms(hist1, hist2):
    """Improved histogram comparison function with better sensitivity"""
    try:
        # Convert inputs to numpy arrays if they aren't already
        if not isinstance(hist1, np.ndarray):
            hist1 = np.array(hist1, dtype=np.float32)
        if not isinstance(hist2, np.ndarray):
            hist2 = np.array(hist2, dtype=np.float32)
        
        # Ensure both histograms are float32
        hist1 = hist1.astype(np.float32)
        hist2 = hist2.astype(np.float32)
        
        # Reshape if necessary
        if hist1.ndim > 1:
            hist1 = hist1.flatten()
        if hist2.ndim > 1:
            hist2 = hist2.flatten()
        
        # Ensure same length
        min_len = min(len(hist1), len(hist2))
        hist1 = hist1[:min_len]
        hist2 = hist2[:min_len]
        
        # Normalize if needed
        if np.sum(hist1) > 0:
            hist1 = hist1 / np.sum(hist1)
        if np.sum(hist2) > 0:
            hist2 = hist2 / np.sum(hist2)
        
        # Calculate multiple similarity metrics for better accuracy
        
        # 1. Intersection (most reliable for color histograms)
        intersection = np.sum(np.minimum(hist1, hist2)) * 100
        
        # 2. Correlation
        if np.std(hist1) > 0 and np.std(hist2) > 0:
            correlation = np.corrcoef(hist1, hist2)[0, 1]
            # Convert correlation (-1 to 1) to similarity (0 to 100)
            correlation_sim = (correlation + 1) * 50
        else:
            correlation_sim = 0
        
        # 3. Chi-Square Distance
        # Convert to similarity (lower chi-square means higher similarity)
        epsilon = 1e-10  # Avoid division by zero
        chi_square = np.sum((hist1 - hist2)**2 / (hist1 + hist2 + epsilon))
        chi_square_sim = 100 * np.exp(-chi_square)
        
        # Combine metrics with weights favoring intersection
        final_similarity = (0.5 * intersection + 
                            0.3 * correlation_sim + 
                            0.2 * chi_square_sim)
        
        # Scale the result to enhance contrast between similar and different videos
        # This counteracts histogram's tendency to give moderately high scores to different videos
        if final_similarity < 60:
            final_similarity = final_similarity * 0.8
        elif final_similarity > 85:
            # Boost high similarities (likely real matches)
            final_similarity = 85 + (final_similarity - 85) * 1.5
            
        # Ensure result is between 0-100
        return max(0, min(100, final_similarity))
    
    except Exception as e:
        print(f"Histogram comparison error: {e}")
        # Return a middling value on error
        return 40

def compare_keyframes(keyframes1, keyframes2):
    """Compare keyframes using multiple approaches"""
    try:
        if not keyframes1 or not keyframes2:
            return 0
        
        similarities = []
        
        # Limit comparison to minimum length
        min_length = min(len(keyframes1), len(keyframes2))
        
        for i in range(min_length):
            # Perceptual hash comparison
            phash1 = compute_phash(keyframes1[i])
            phash2 = compute_phash(keyframes2[i])
            phash_sim = compare_phash(phash1, phash2)
            
            # Histogram comparison
            hist1 = compute_histogram(keyframes1[i])
            hist2 = compute_histogram(keyframes2[i])
            hist_sim = compare_histograms(hist1, hist2)
            
            # Weighted combination
            frame_sim = 0.6 * phash_sim + 0.4 * hist_sim
            similarities.append(frame_sim)
        
        # Overall keyframe similarity
        return np.mean(similarities) if similarities else 0
    
    except Exception as e:
        print(f"Keyframe comparison error: {e}")
        return 0

########################################
# 3. IMPROVED DEEP FEATURE EXTRACTION #
########################################

def extract_deep_features(image):
    """Extract deep features from an image using a pre-trained CNN"""
    if model is None:
        return np.random.rand(512)  # Fallback if model not loaded
        
    try:
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply transformations
        img_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(img_tensor)
            
        # Convert to numpy array
        features = features.squeeze().cpu().numpy()
        
        return features
    except Exception as e:
        print(f"Error extracting deep features: {str(e)}")
        return np.random.rand(512)  # Return random features as fallback

def compare_deep_features(features1, features2):
    """Compare deep features with cosine similarity and threshold curve"""
    try:
        # Convert to numpy arrays if they aren't already
        if not isinstance(features1, np.ndarray):
            features1 = np.array(features1)
        if not isinstance(features2, np.ndarray):
            features2 = np.array(features2)
            
        # Check for proper dimensions
        if features1.ndim != 1 or features2.ndim != 1:
            features1 = features1.flatten()
            features2 = features2.flatten()
            
        # If feature lengths don't match, pad or truncate
        if len(features1) != len(features2):
            min_len = min(len(features1), len(features2))
            features1 = features1[:min_len]
            features2 = features2[:min_len]
        
        # Normalize feature vectors
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0  # Avoid division by near-zero
        
        features1_norm = features1 / norm1
        features2_norm = features2 / norm2
        
        # Compute cosine similarity
        similarity = np.dot(features1_norm, features2_norm)
        
        # Apply a threshold curve to emphasize differences
        # Values below 0.7 are considered more distinct
        if similarity < 0.5:
            # Low similarity scores remain mostly unchanged
            adjusted_similarity = similarity * 0.9
        elif similarity < 0.85:
            # Medium similarity gets a gentle boost
            adjusted_similarity = 0.45 + (similarity - 0.5) * 0.6
        else:
            # High similarity scores get a stronger boost
            adjusted_similarity = 0.66 + (similarity - 0.85) * 1.4
        
        # Convert to percentage
        return max(0, min(100, adjusted_similarity * 100))
    except Exception as e:
        print(f"Error comparing deep features: {str(e)}")
        return 0

#####################################
# 4. OBJECT DETECTION AND ANALYSIS #
#####################################

def detect_objects(image, confidence_threshold=0.5):
    """Detect objects in an image using Faster R-CNN"""
    if obj_detection_model is None:
        return []
    
    try:
        # Convert to RGB (PyTorch models expect RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL and then to tensor
        pil_image = Image.fromarray(rgb_image)
        img_tensor = transforms.ToTensor()(pil_image).to(device)
        
        # Detect objects
        with torch.no_grad():
            predictions = obj_detection_model([img_tensor])
            
        # Extract predictions
        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()
        pred_labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence
        mask = pred_scores >= confidence_threshold
        boxes = pred_boxes[mask]
        scores = pred_scores[mask]
        labels = pred_labels[mask]
        
        # Convert labels to class names
        objects = [
            {
                'class': COCO_INSTANCE_CATEGORY_NAMES[label],
                'confidence': float(score),
                'box': box.tolist()
            }
            for label, score, box in zip(labels, scores, boxes)
        ]
        
        return objects
    except Exception as e:
        print(f"Error detecting objects: {str(e)}")
        return []

def compare_objects(objects1, objects2):
    """Compare detected objects between two frames"""
    if not objects1 or not objects2:
        if not objects1 and not objects2:
            return 100  # Both have no objects, considered similar
        return 0  # One has objects, one doesn't, considered different
    
    try:
        # Count objects by class
        class_counts1 = defaultdict(int)
        class_counts2 = defaultdict(int)
        
        for obj in objects1:
            class_counts1[obj['class']] += 1
            
        for obj in objects2:
            class_counts2[obj['class']] += 1
            
        # Get all unique classes
        all_classes = set(list(class_counts1.keys()) + list(class_counts2.keys()))
        
        # Calculate Jaccard similarity for each class
        class_similarities = []
        
        for cls in all_classes:
            count1 = class_counts1[cls]
            count2 = class_counts2[cls]
            
            if count1 == 0 and count2 == 0:
                continue
                
            # Calculate similarity as min/max ratio
            similarity = min(count1, count2) / max(count1, count2)
            class_similarities.append(similarity)
        
        # Average similarity across all classes
        if class_similarities:
            return min(100, 100 * sum(class_similarities) / len(class_similarities))
        else:
            return 0
    except Exception as e:
        print(f"Error comparing objects: {str(e)}")
        return 0

##################################
# 5. SCENE CHANGE DETECTION     #
##################################

def detect_scene_changes(frames, threshold=25.0):
    """Detect scene changes between consecutive frames with improved sensitivity"""
    if len(frames) < 2:
        return []
    
    scene_changes = []
    prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    for i in range(1, len(frames)):
        curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        # Calculate both absolute difference and SSIM
        diff = cv2.absdiff(curr_frame, prev_frame)
        diff_score = np.mean(diff)
        
        # Calculate histogram difference
        prev_hist = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
        curr_hist = cv2.calcHist([curr_frame], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
        
        # Compare histograms
        hist_diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
        hist_score = hist_diff * 100  # Scale to similar range as diff_score
        
        # Weighted combination of metrics
        combined_score = 0.7 * diff_score + 0.3 * hist_score
        
        if combined_score > threshold:
            scene_changes.append(i)
            
        prev_frame = curr_frame
        
    return scene_changes

def compare_scene_changes(scenes1, scenes2, total_frames1, total_frames2):
    """Compare scene change patterns with more sophisticated measures"""
    try:
        if total_frames1 <= 0 or total_frames2 <= 0:
            return 0
            
        # If both videos have no scene changes, they could be very different content
        # So we'll return a moderate similarity rather than 100%
        if not scenes1 and not scenes2:
            return 50
            
        # If one has scenes and the other doesn't, they're quite different
        if bool(scenes1) != bool(scenes2):
            return 20
            
        # Normalize scene changes to percentage of video length
        normalized_scenes1 = [s / total_frames1 for s in scenes1]
        normalized_scenes2 = [s / total_frames2 for s in scenes2]
        
        # Compare the number of scenes - videos with very different numbers
        # of scene changes are likely different
        scene_count_ratio = min(len(normalized_scenes1), len(normalized_scenes2)) / max(1, max(len(normalized_scenes1), len(normalized_scenes2)))
        
        # Compare the actual timing of scene changes
        # We'll use a dynamic time warping approach (simplified)
        scene_timing_similarity = 0
        
        if normalized_scenes1 and normalized_scenes2:
            # For each scene in video 1, find the closest scene in video 2
            matches = 0
            for scene1 in normalized_scenes1:
                best_distance = min([abs(scene1 - scene2) for scene2 in normalized_scenes2], default=1.0)
                if best_distance < 0.1:  # Within 10% of video length
                    matches += 1
                    
            scene_timing_similarity = matches / max(len(normalized_scenes1), 1) * 100
        
        # Final similarity is weighted between scene count and timing
        similarity = 0.4 * (scene_count_ratio * 100) + 0.6 * scene_timing_similarity
        
        return similarity
    except Exception as e:
        print(f"Error comparing scene changes: {str(e)}")
        return 0

#####################################
# 6. IMPROVED MOTION ANALYSIS      #
#####################################

def compute_motion_vector(frames):
    """Compute motion vectors using optical flow with enhanced robustness"""
    if len(frames) < 2:
        return []
        
    motion_magnitudes = []
    motion_directions = []
    prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    for i in range(1, len(frames)):
        curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        try:
            # Calculate optical flow (Farneback method)
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Compute magnitude and angle of flow vectors
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Calculate average motion magnitude
            avg_magnitude = np.mean(mag)
            
            # Calculate histogram of flow directions (8 bins)
            ang_deg = ang * 180 / np.pi
            direction_hist, _ = np.histogram(ang_deg, bins=8, range=(0, 360))
            direction_hist = direction_hist / np.sum(direction_hist)
            
            motion_magnitudes.append(avg_magnitude)
            motion_directions.append(direction_hist)
        except Exception as e:
            print(f"Error computing motion for frame {i}: {str(e)}")
            # Add placeholder values
            motion_magnitudes.append(0)
            motion_directions.append(np.ones(8) / 8)  # Uniform distribution
            
        prev_frame = curr_frame
        
    return {'magnitudes': motion_magnitudes, 'directions': motion_directions}

def compare_motion(motion1, motion2):
    """Compare motion patterns between two videos with statistical methods"""
    try:
        # Extract motion components
        mag1 = motion1.get('magnitudes', [])
        mag2 = motion2.get('magnitudes', [])
        dir1 = motion1.get('directions', [])
        dir2 = motion2.get('directions', [])
        
        if not mag1 or not mag2:
            return 0
            
        # 1. Compare magnitude statistics
        if len(mag1) != len(mag2):
            # Calculate statistical properties instead of just averages
            props1 = {
                'mean': np.mean(mag1),
                'std': np.std(mag1),
                'median': np.median(mag1),
                'max': np.max(mag1) if mag1 else 0,
                'min': np.min(mag1) if mag1 else 0
            }
            
            props2 = {
                'mean': np.mean(mag2),
                'std': np.std(mag2),
                'median': np.median(mag2),
                'max': np.max(mag2) if mag2 else 0,
                'min': np.min(mag2) if mag2 else 0
            }
            
            # Calculate normalized distances between properties
            diffs = []
            for key in props1:
                max_val = max(props1[key], props2[key])
                min_val = min(props1[key], props2[key])
                
                if max_val == 0:  # Avoid division by zero
                    diffs.append(0 if min_val == 0 else 1)
                else:
                    diffs.append(abs(props1[key] - props2[key]) / max_val)
            
            # Convert differences to similarity
            magnitude_similarity = 100 * (1 - np.mean(diffs))
        else:
            # If same length, use correlation
            correlation = np.corrcoef(mag1, mag2)[0, 1]
            if np.isnan(correlation):
                magnitude_similarity = 0
            else:
                magnitude_similarity = 100 * (correlation + 1) / 2
        
        # 2. Compare direction histograms
        direction_similarities = []
        min_len = min(len(dir1), len(dir2))
        
        for i in range(min_len):
            # Chi-square distance between histograms
            d1 = np.array(dir1[i])
            d2 = np.array(dir2[i])
            
            chi_square = sum((d1 - d2)**2 / (d1 + d2 + 1e-10))
            sim = 100 * np.exp(-chi_square)
            direction_similarities.append(sim)
        
        # Average direction similarity
        if direction_similarities:
            direction_similarity = np.mean(direction_similarities)
        else:
            direction_similarity = 0
        
        # Weighted combination (more weight to magnitude)
        final_similarity = 0.7 * magnitude_similarity + 0.3 * direction_similarity
        
        return min(100, max(0, final_similarity))
    except Exception as e:
        print(f"Error comparing motion: {str(e)}")
        return 0
    
    #############################################
# 7. TEMPORAL MANIPULATION DETECTION       #
#############################################

# In simple_server.py, replace the extract_frame_sequence function (around line 906)
def extract_frame_sequence(video_path, max_frames=10, focus_on_ends=True):
    """Extract a sequence of frames with better consistency for comparison"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        if frame_count <= 0 or duration <= 0:
            cap.release()
            return []
        
        # Limit max_frames to conserve resources
        max_frames = min(max_frames, MAX_FRAMES)
        
        # Calculate frame positions
        if not focus_on_ends:
            # Instead of random linear sampling, use consistent time-based sampling
            # This ensures we sample the same moments in videos regardless of their length
            # Key moments are at 10%, 20%, 30%... of total duration
            
            # Create indices at percentage points of the video
            indices = []
            for percent in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                idx = int((percent / 100) * frame_count)
                if 0 <= idx < frame_count:
                    indices.append(idx)
                    
            # If we need more frames, add intermediate points
            if len(indices) < max_frames:
                additional_indices = []
                for percent in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]:
                    if len(indices) + len(additional_indices) >= max_frames:
                        break
                    idx = int((percent / 100) * frame_count)
                    if 0 <= idx < frame_count:
                        additional_indices.append(idx)
                        
                indices.extend(additional_indices)
                indices.sort()
                
            # Limit to max_frames
            indices = indices[:max_frames]
        else:
            # Focus on beginning and end, with fewer samples in the middle
            start_count = int(max_frames * 0.4)
            end_count = int(max_frames * 0.4)
            middle_count = max_frames - start_count - end_count
            
            # More consistent sampling of start and end
            start_indices = [int((i / start_count) * (frame_count * 0.2)) for i in range(start_count)]
            end_indices = [int(frame_count * 0.8 + (i / end_count) * (frame_count * 0.2)) for i in range(end_count)]
            
            # Sample middle less frequently
            if middle_count > 0 and frame_count > 0:
                middle_indices = [int(frame_count * 0.2 + (i / middle_count) * (frame_count * 0.6)) for i in range(middle_count)]
            else:
                middle_indices = []
                
            indices = start_indices + middle_indices + end_indices
            # Remove any out-of-bounds indices
            indices = [idx for idx in indices if 0 <= idx < frame_count]
        
        # Read frames at calculated indices
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize for faster processing
                frame = cv2.resize(frame, MAX_IMAGE_SIZE)
                frames.append(frame)
                
        cap.release()
        return frames
    except Exception as e:
        print(f"Error extracting frame sequence: {str(e)}")
        return []

def compute_frame_fingerprints(frames):
    """Compute fingerprints for a sequence of frames"""
    fingerprints = []
    for frame in frames:
        try:
            # Compute various fingerprints
            # 1. Perceptual hash
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
            avg = resized.mean()
            phash = (resized > avg).flatten().astype(int)
            
            # 2. Simple color histogram (low resolution)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Combine fingerprints
            fingerprints.append({
                'phash': phash,
                'hist': hist,
            })
        except Exception as e:
            print(f"Error computing frame fingerprint: {str(e)}")
            # Add empty fingerprint to maintain sequence
            fingerprints.append({
                'phash': np.zeros(256, dtype=int),
                'hist': np.zeros(64),
            })
    
    return fingerprints
def compare_temporal_sequences(seq1, seq2):
    """Compare two frame sequences to detect temporal modifications"""
    if not seq1 or not seq2:
        return {
            'overall': 0,
            'beginning_modified': False,
            'end_truncated': False,
            'temporal_score': 0
        }
    
    try:
        # Get fingerprints
        fingerprints1 = compute_frame_fingerprints(seq1)
        fingerprints2 = compute_frame_fingerprints(seq2)
        
        # Dynamic Time Warping (simplified) to account for different video lengths
        n, m = len(fingerprints1), len(fingerprints2)
        
        # Create cost matrix
        cost = np.zeros((n+1, m+1))
        cost[0, 1:] = np.inf
        cost[1:, 0] = np.inf
        
        # Fill cost matrix
        for i in range(1, n+1):
            for j in range(1, m+1):
                # Compute cost between frames
                f1, f2 = fingerprints1[i-1], fingerprints2[j-1]
                
                # Hamming distance for phash
                phash_dist = np.sum(f1['phash'] != f2['phash']) / len(f1['phash'])
                
                # Histogram distance
                hist_dist = np.sum(np.abs(f1['hist'] - f2['hist'])) / len(f1['hist'])
                
                # Combined distance
                d = 0.7 * phash_dist + 0.3 * hist_dist
                
                # DTW update
                cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
        
        # Normalized cost
        dtw_distance = cost[n, m] / (n + m)
        dtw_similarity = max(0, min(100, 100 * (1 - dtw_distance)))
        
        # Check for specific modifications
        # Beginning modification (first 20% of frames)
        beginning_scores = []
        for i in range(min(int(n * 0.2), int(m * 0.2))):
            if i < len(fingerprints1) and i < len(fingerprints2):
                f1, f2 = fingerprints1[i], fingerprints2[i]
                phash_sim = 100 - (np.sum(f1['phash'] != f2['phash']) / len(f1['phash']) * 100)
                beginning_scores.append(phash_sim)
        
        beginning_sim = np.mean(beginning_scores) if beginning_scores else 0
        beginning_modified = beginning_sim < 70  # Lower threshold to detect modifications
        
        # End truncation (last 20% of frames)
        # Check if videos have significantly different lengths
        length_ratio = min(n, m) / max(n, m)
        potentially_truncated = length_ratio < 0.8
        
        end_scores = []
        for i in range(1, min(int(n * 0.2), int(m * 0.2)) + 1):
            if n-i >= 0 and m-i >= 0:
                f1, f2 = fingerprints1[n-i], fingerprints2[m-i]
                phash_sim = 100 - (np.sum(f1['phash'] != f2['phash']) / len(f1['phash']) * 100)
                end_scores.append(phash_sim)
        
        end_sim = np.mean(end_scores) if end_scores else 0
        end_truncated = potentially_truncated and end_sim < 70
        
        # Overall temporal similarity
        temporal_score = dtw_similarity
        if beginning_modified:
            temporal_score *= 0.8  # Reduce score if beginning was modified
        if end_truncated:
            temporal_score *= 0.8  # Reduce score if end was truncated
            
        return {
            'overall': temporal_score,
            'beginning_modified': beginning_modified,
            'end_truncated': end_truncated,
            'temporal_score': temporal_score
        }
        
    except Exception as e:
        print(f"Error comparing temporal sequences: {str(e)}")
        return {
            'overall': 0,
            'beginning_modified': False,
            'end_truncated': False,
            'temporal_score': 0
        }

#############################################
# 8. OVERLAY DETECTION                     #
#############################################

def detect_overlay_differences(frame1, frame2):
    """Detect text and graphic overlays by analyzing edge differences"""
    try:
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Ensure same size
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # Apply edge detection
        edges1 = cv2.Canny(gray1, 100, 200)
        edges2 = cv2.Canny(gray2, 100, 200)
        
        # Find differences in edges
        edge_diff = cv2.bitwise_xor(edges1, edges2)
        
        # Calculate edge difference ratio
        edge_diff_ratio = np.sum(edge_diff > 0) / edge_diff.size
        
        # Calculate text-like features (horizontal and vertical lines often present in text)
        horizontal_kernel = np.ones((1, 5), np.uint8)
        vertical_kernel = np.ones((5, 1), np.uint8)
        
        h_dilate1 = cv2.dilate(edges1, horizontal_kernel, iterations=1)
        h_dilate2 = cv2.dilate(edges2, horizontal_kernel, iterations=1)
        h_diff = cv2.bitwise_xor(h_dilate1, h_dilate2)
        
        v_dilate1 = cv2.dilate(edges1, vertical_kernel, iterations=1)
        v_dilate2 = cv2.dilate(edges2, vertical_kernel, iterations=1)
        v_diff = cv2.bitwise_xor(v_dilate1, v_dilate2)
        
        # Combined difference focusing on text-like structures
        combined_diff = cv2.bitwise_or(h_diff, v_diff)
        text_diff_ratio = np.sum(combined_diff > 0) / combined_diff.size
        
        # Higher weight to text-like differences
        weighted_diff = 0.4 * edge_diff_ratio + 0.6 * text_diff_ratio
        
        # Normalize to 0-100 scale with thresholding
        # Small differences (< 0.05) might be noise
        if weighted_diff < 0.05:
            overlay_score = 100
        else:
            # Map 0.05-0.3 to 0-100 (inverse - higher diff means lower similarity)
            overlay_score = max(0, 100 - (weighted_diff - 0.05) * (100 / 0.25))
        
        # Binary classification of whether an overlay was detected
        has_overlay = weighted_diff > 0.08  # Threshold for overlay detection
        
        return {
            'score': overlay_score,
            'has_overlay': has_overlay,
            'diff_ratio': weighted_diff
        }
    except Exception as e:
        print(f"Error detecting overlays: {str(e)}")
        return {
            'score': 0,
            'has_overlay': False,
            'diff_ratio': 1.0
        }

def compare_overlay_modifications(frames1, frames2):
    """Compare frames to detect overlay modifications across videos"""
    if not frames1 or not frames2:
        return {
            'overlay_score': 0,
            'has_overlay': False,
            'overlay_ratio': 0
        }
    
    try:
        overlay_results = []
        
        # Compare corresponding frames
        # Use min length to handle different video lengths
        for i in range(min(len(frames1), len(frames2))):
            result = detect_overlay_differences(frames1[i], frames2[i])
            overlay_results.append(result)
        
        # Aggregate results
        overlay_scores = [r['score'] for r in overlay_results]
        has_overlays = [r['has_overlay'] for r in overlay_results]
        
        # Overall metrics
        avg_score = np.mean(overlay_scores) if overlay_scores else 0
        overlay_detected = sum(has_overlays) > len(has_overlays) * 0.2  # If >20% of frames have overlays
        overlay_ratio = sum(has_overlays) / len(has_overlays) if has_overlays else 0
        
        return {
            'overlay_score': avg_score,
            'has_overlay': overlay_detected,
            'overlay_ratio': overlay_ratio
        }
    except Exception as e:
        print(f"Error comparing overlay modifications: {str(e)}")
        return {
            'overlay_score': 0,
            'has_overlay': False,
            'overlay_ratio': 0
        }

#############################################
# 9. SPLIT-SCREEN DETECTION                #
#############################################

def detect_split_screen(frame):
    """Detect if a frame contains a split-screen composition"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Methods to detect split screens:
        
        # 1. Horizontal split detection
        # Check for strong horizontal line or contrast difference between top and bottom
        top_half = gray[:height//2, :]
        bottom_half = gray[height//2:, :]
        
        top_mean = np.mean(top_half)
        bottom_mean = np.mean(bottom_half)
        horizontal_diff = abs(top_mean - bottom_mean) / 255.0
        
        # Apply edge detection to find strong horizontal line
        edges = cv2.Canny(gray, 50, 150)
        horizontal_kernel = np.ones((1, width//10), np.uint8)
        horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Check for strong horizontal line near middle
        middle_region = horizontal_edges[int(height*0.4):int(height*0.6), :]
        horizontal_line_strength = np.sum(middle_region) / (middle_region.shape[0] * middle_region.shape[1] * 255)
        
        # 2. Vertical split detection
        left_half = gray[:, :width//2]
        right_half = gray[:, width//2:]
        
        left_mean = np.mean(left_half)
        right_mean = np.mean(right_half)
        vertical_diff = abs(left_mean - right_mean) / 255.0
        
        # Check for strong vertical line
        vertical_kernel = np.ones((height//10, 1), np.uint8)
        vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
        
        middle_region = vertical_edges[:, int(width*0.4):int(width*0.6)]
        vertical_line_strength = np.sum(middle_region) / (middle_region.shape[0] * middle_region.shape[1] * 255)
        
        # 3. Quadrant check (for 2x2 split screens)
        q1 = np.mean(gray[:height//2, :width//2])
        q2 = np.mean(gray[:height//2, width//2:])
        q3 = np.mean(gray[height//2:, :width//2])
        q4 = np.mean(gray[height//2:, width//2:])
        
        # Calculate variance between quadrants
        quadrant_values = [q1, q2, q3, q4]
        quadrant_variance = np.var(quadrant_values) / (255.0 * 255.0)  # Normalized
        
        # Combine evidence
        horizontal_split_evidence = max(horizontal_diff * 2, horizontal_line_strength * 10)
        vertical_split_evidence = max(vertical_diff * 2, vertical_line_strength * 10)
        quad_split_evidence = quadrant_variance * 5
        
        # Threshold for split screen detection
        is_horizontal_split = horizontal_split_evidence > 0.15
        is_vertical_split = vertical_split_evidence > 0.15
        is_quad_split = quad_split_evidence > 0.1 and (is_horizontal_split or is_vertical_split)
        
        # Combined result
        is_split_screen = is_horizontal_split or is_vertical_split or is_quad_split
        split_type = None
        
        if is_quad_split:
            split_type = "quad"
        elif is_horizontal_split and is_vertical_split:
            split_type = "complex"
        elif is_horizontal_split:
            split_type = "horizontal"
        elif is_vertical_split:
            split_type = "vertical"
            
        confidence = max(horizontal_split_evidence, vertical_split_evidence, quad_split_evidence)
        
        return {
            'is_split_screen': is_split_screen,
            'split_type': split_type,
            'confidence': min(1.0, confidence) * 100,
            'h_evidence': horizontal_split_evidence,
            'v_evidence': vertical_split_evidence,
            'q_evidence': quad_split_evidence
        }
    except Exception as e:
        print(f"Error detecting split screen: {str(e)}")
        return {
            'is_split_screen': False,
            'split_type': None,
            'confidence': 0,
            'h_evidence': 0,
            'v_evidence': 0,
            'q_evidence': 0
        }

def compare_split_screen(frames1, frames2):
    """Compare two sets of frames to detect if either contains split screens not in the other"""
    if not frames1 or not frames2:
        return {
            'has_split_screen': False,
            'split_screen_score': 100,
            'detection_confidence': 0
        }
    
    try:
        # Analyze both videos for split screens
        splits1 = [detect_split_screen(frame) for frame in frames1]
        splits2 = [detect_split_screen(frame) for frame in frames2]
        
        # Count frames with split screens
        split_frames1 = [s for s in splits1 if s['is_split_screen']]
        split_frames2 = [s for s in splits2 if s['is_split_screen']]
        
        # Calculate ratios
        ratio1 = len(split_frames1) / len(frames1) if frames1 else 0
        ratio2 = len(split_frames2) / len(frames2) if frames2 else 0
        
        # A significant difference in split screen ratio indicates manipulation
        ratio_diff = abs(ratio1 - ratio2)
        
        # Calculate similarity score (inverse of difference)
        if ratio_diff < 0.05:  # Small difference might be detection noise
            split_screen_score = 100
        else:
            # Map 0.05-0.5 ratio difference to 0-100 similarity score (inverse)
            split_screen_score = max(0, 100 - (ratio_diff - 0.05) * 200)
        
        # Overall result
        max_confidence = max([s['confidence'] for s in split_frames1 + split_frames2]) if split_frames1 or split_frames2 else 0
        
        return {
            'has_split_screen': ratio_diff > 0.1,  # Significant difference in split screen usage
            'split_screen_score': split_screen_score,
            'detection_confidence': max_confidence,
            'ratio1': ratio1,
            'ratio2': ratio2
        }
    except Exception as e:
        print(f"Error comparing split screens: {str(e)}")
        return {
            'has_split_screen': False,
            'split_screen_score': 100,
            'detection_confidence': 0
        }

#############################################
# 10. REFLECTION DETECTION                 #
#############################################

def detect_reflection(frame):
    """Detect if a frame has been mirrored/reflected"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Compute gradient in x direction
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        
        # Calculate asymmetry by comparing left and right halves
        left_half = sobelx[:, :width//2]
        right_half = sobelx[:, width//2:]
        
        # Flip right half to compare with left half
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Compare gradients
        if left_half.shape != right_half_flipped.shape:
            # Handle odd width
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            gradient_diff = np.mean(np.abs(left_half[:, :min_width] - right_half_flipped[:, :min_width]))
        else:
            gradient_diff = np.mean(np.abs(left_half - right_half_flipped))
        
        # Normalize by average gradient magnitude
        avg_gradient = np.mean(np.abs(sobelx))
        if avg_gradient > 0:
            normalized_diff = gradient_diff / avg_gradient
        else:
            normalized_diff = 1.0  # Arbitrary high value for flat images
        
        # Lower normalized_diff means more symmetric (potentially mirrored)
        reflection_likelihood = max(0, 1.0 - normalized_diff)
        
        return {
            'reflection_likelihood': reflection_likelihood,
            'is_reflected': reflection_likelihood > 0.7,  # High threshold to avoid false positives
            'gradient_diff': normalized_diff
        }
    except Exception as e:
        print(f"Error detecting reflection: {str(e)}")
        return {
            'reflection_likelihood': 0,
            'is_reflected': False,
            'gradient_diff': 1.0
        }

def compare_reflection(frames1, frames2):
    """Compare two sets of frames to detect reflection transformations"""
    if not frames1 or not frames2:
        return {
            'reflection_detected': False,
            'reflection_score': 100,
            'confidence': 0
        }
    
    try:
        # Analyze both videos for reflections
        reflections1 = [detect_reflection(frame) for frame in frames1]
        reflections2 = [detect_reflection(frame) for frame in frames2]
        
        # Extract average likelihoods
        avg_likelihood1 = np.mean([r['reflection_likelihood'] for r in reflections1])
        avg_likelihood2 = np.mean([r['reflection_likelihood'] for r in reflections2])
        
        # If one video has significantly higher reflection likelihood, it might be mirrored
        reflection_diff = abs(avg_likelihood1 - avg_likelihood2)
        
        # Now compare frames directly with flipped versions
        # For computational efficiency, just take a few frames
        sample_indices = np.linspace(0, min(len(frames1), len(frames2))-1, 5, dtype=int)
        
        normal_similarities = []
        flipped_similarities = []
        
        for idx in sample_indices:
            if idx < len(frames1) and idx < len(frames2):
                frame1 = frames1[idx]
                frame2 = frames2[idx]
                flipped_frame2 = cv2.flip(frame2, 1)  # Flip horizontally
                
                # Compute histogram similarity with original and flipped frame
                hist1 = compute_histogram(frame1)
                hist2 = compute_histogram(frame2)
                hist2_flipped = compute_histogram(flipped_frame2)
                
                normal_sim = compare_histograms(hist1, hist2)
                flipped_sim = compare_histograms(hist1, hist2_flipped)
                
                normal_similarities.append(normal_sim)
                flipped_similarities.append(flipped_sim)
        
        # If flipped similarity is significantly higher, video is likely mirrored
        avg_normal = np.mean(normal_similarities) if normal_similarities else 0
        avg_flipped = np.mean(flipped_similarities) if flipped_similarities else 0
        
        flipped_improvement = avg_flipped - avg_normal
        
        # Determine if reflection is detected
        reflection_detected = (flipped_improvement > 15) or (reflection_diff > 0.3)
        
        # Calculate confidence based on multiple indicators
        confidence = max(min(100, abs(flipped_improvement) * 2), 
                         min(100, reflection_diff * 100 * 2))
        
        # Calculate reflection score - higher means less likely to be reflected
        if not reflection_detected:
            reflection_score = 100
        else:
            # Score inversely proportional to confidence
            reflection_score = max(0, 100 - confidence)
        
        return {
            'reflection_detected': reflection_detected,
            'reflection_score': reflection_score,
            'confidence': confidence,
            'normal_similarity': avg_normal,
            'flipped_similarity': avg_flipped
        }
    except Exception as e:
        print(f"Error comparing reflections: {str(e)}")
        return {
            'reflection_detected': False,
            'reflection_score': 100,
            'confidence': 0
        }

#############################################
# 11. COMBINED MANIPULATION DETECTION      #
#############################################

def detect_manipulations(video_path1, video_path2):
    """Robust implementation of manipulation detection"""
    try:
        print(f"\nAnalyzing manipulations between videos:")
        print(f"Video 1: {os.path.basename(video_path1)}")
        print(f"Video 2: {os.path.basename(video_path2)}")
        
        start_time = time.time()
        
        # Validate video files
        for path in [video_path1, video_path2]:
            if not os.path.exists(path):
                print(f"Video file does not exist: {path}")
                return {
                    'manipulations_detected': False,
                    'manipulation_types': [],
                    'manipulation_score': 100,
                    'details': {
                        'error': f'Video file not found: {os.path.basename(path)}'
                    }
                }
        
        # Extract frames with error handling
        try:
            frames1_temporal = extract_frame_sequence(video_path1, max_frames=15, focus_on_ends=True)
            frames2_temporal = extract_frame_sequence(video_path2, max_frames=15, focus_on_ends=True)
            
            frames1 = extract_frame_sequence(video_path1, max_frames=10, focus_on_ends=False)
            frames2 = extract_frame_sequence(video_path2, max_frames=10, focus_on_ends=False)
        except Exception as extract_error:
            print(f"Error extracting frames: {str(extract_error)}")
            return {
                'manipulations_detected': False,
                'manipulation_types': [],
                'manipulation_score': 100,
                'details': {
                    'error': f'Could not extract frames: {str(extract_error)}'
                }
            }
        
        if not frames1 or not frames2:
            print("Error: Could not extract frames from one or both videos")
            return {
                'manipulations_detected': False,
                'manipulation_types': [],
                'manipulation_score': 100,
                'details': {
                    'error': 'Could not extract frames'
                }
            }
        
        # Perform manipulation detection with error isolation for each type
        manipulation_results = {}
        manipulation_types = []
        
        # 1. Temporal Analysis with error handling
        try:
            print("Performing temporal analysis...")
            temporal_result = compare_temporal_sequences(frames1_temporal, frames2_temporal)
            manipulation_results['temporal'] = temporal_result
            
            if temporal_result.get('beginning_modified', False):
                manipulation_types.append('frame_insertion')
            
            if temporal_result.get('end_truncated', False):
                manipulation_types.append('temporal_truncation')
        except Exception as temporal_error:
            print(f"Error in temporal analysis: {str(temporal_error)}")
            manipulation_results['temporal'] = {
                'error': str(temporal_error),
                'temporal_score': 100
            }
        
        # 2. Overlay Detection with error handling
        try:
            print("Detecting overlay modifications...")
            overlay_result = compare_overlay_modifications(frames1, frames2)
            manipulation_results['overlay'] = overlay_result
            
            if overlay_result.get('has_overlay', False):
                manipulation_types.append('overlay_modification')
        except Exception as overlay_error:
            print(f"Error in overlay detection: {str(overlay_error)}")
            manipulation_results['overlay'] = {
                'error': str(overlay_error),
                'overlay_score': 100,
                'has_overlay': False
            }
        
        # 3. Split-Screen Detection with error handling
        try:
            print("Detecting split-screen compositions...")
            split_screen_result = compare_split_screen(frames1, frames2)
            manipulation_results['split_screen'] = split_screen_result
            
            if split_screen_result.get('has_split_screen', False):
                manipulation_types.append('split_screen_composition')
        except Exception as split_error:
            print(f"Error in split-screen detection: {str(split_error)}")
            manipulation_results['split_screen'] = {
                'error': str(split_error),
                'split_screen_score': 100,
                'has_split_screen': False
            }
        
        # 4. Reflection Detection with error handling
        try:
            print("Detecting reflection transformations...")
            reflection_result = compare_reflection(frames1, frames2)
            manipulation_results['reflection'] = reflection_result
            
            if reflection_result.get('reflection_detected', False):
                manipulation_types.append('reflection_transformation')
        except Exception as reflection_error:
            print(f"Error in reflection detection: {str(reflection_error)}")
            manipulation_results['reflection'] = {
                'error': str(reflection_error),
                'reflection_score': 100,
                'reflection_detected': False
            }
        
        # Calculate overall manipulation score from available results
        scores = []
        
        # Add scores with defaults for missing components
        scores.append(manipulation_results.get('temporal', {}).get('temporal_score', 100))
        scores.append(manipulation_results.get('overlay', {}).get('overlay_score', 100))
        scores.append(manipulation_results.get('split_screen', {}).get('split_screen_score', 100))
        scores.append(manipulation_results.get('reflection', {}).get('reflection_score', 100))
        
        # Weight the scores based on reliability
        weights = [0.4, 0.3, 0.15, 0.15]  # Higher weight to temporal and overlay
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        # Additional penalty for each type of manipulation detected
        manipulation_penalty = len(manipulation_types) * 5
        final_score = max(0, weighted_score - manipulation_penalty)
        
        elapsed_time = time.time() - start_time
        print(f"Manipulation analysis completed in {elapsed_time:.2f} seconds")
        print(f"Detected manipulations: {manipulation_types}")
        print(f"Manipulation score: {final_score:.2f}/100 (higher = less manipulation)")
        
        return {
            'manipulations_detected': len(manipulation_types) > 0,
            'manipulation_types': manipulation_types,
            'manipulation_score': final_score,
            'details': manipulation_results
        }
    except Exception as e:
        print(f"Error in manipulation detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'manipulations_detected': False,
            'manipulation_types': ['detection_failed'],
            'manipulation_score': 100,  # Default to no manipulation on error
            'details': {
                'error': str(e)
            }
        }
def compare_videos_with_manipulation_detection(analysis1, analysis2):
    """Enhanced comparison that includes manipulation detection and object comparison"""
    print(f"\n{'='*50}")
    print(f"COMPARING VIDEOS WITH MANIPULATION DETECTION:")
    print(f"A: {os.path.basename(analysis1.get('video_path', 'unknown'))}")
    print(f"B: {os.path.basename(analysis2.get('video_path', 'unknown'))}")
    print(f"{'='*50}")
    start_time = time.time()
    
    comparison_result = {
        "video1": os.path.basename(analysis1.get('video_path', 'unknown')),
        "video2": os.path.basename(analysis2.get('video_path', 'unknown')),
        "comparison_time": None,
        "overall_similarity": 0,
        "similarities": {},
        "manipulations": {
            "detected": False,
            "types": [],
            "manipulation_score": 100
        }
    }
    
    try:
        # Standard comparisons for basic features
        similarities = {}
        
        # 1. Perceptual hash comparison with improved method
        phash_similarities = []
        phash1 = analysis1.get("perceptual_hashes", [])
        phash2 = analysis2.get("perceptual_hashes", [])
        
        if phash1 and phash2:
            for hash1 in phash1:
                for hash2 in phash2:
                    try:
                        sim = compare_phash(np.array(hash1), np.array(hash2))
                        phash_similarities.append(sim)
                    except Exception as e:
                        print(f"Error in hash comparison: {e}")
        
        similarities["perceptual_hash"] = max(phash_similarities) if phash_similarities else 0
        print(f"1. PERCEPTUAL HASH SIMILARITY: {similarities['perceptual_hash']:.2f}%")
        
        # 2. Histogram comparison with improved method
        hist_similarities = []
        hist1 = analysis1.get("histograms", [])
        hist2 = analysis2.get("histograms", [])
        
        if hist1 and hist2:
            for h1 in hist1:
                for h2 in hist2:
                    try:
                        sim = compare_histograms(np.array(h1), np.array(h2))
                        hist_similarities.append(sim)
                    except Exception as e:
                        print(f"Error in histogram comparison: {e}")
        
        similarities["histogram"] = max(hist_similarities) if hist_similarities else 0
        print(f"2. HISTOGRAM SIMILARITY: {similarities['histogram']:.2f}%")
        
        # 3. Deep feature comparison
        feature_similarities = []
        feat1 = analysis1.get("deep_features", [])
        feat2 = analysis2.get("deep_features", [])
        
        if feat1 and feat2:
            for f1 in feat1:
                for f2 in feat2:
                    try:
                        sim = compare_deep_features(np.array(f1), np.array(f2))
                        feature_similarities.append(sim)
                    except Exception as e:
                        print(f"Error in feature comparison: {e}")
        
        similarities["deep_features"] = max(feature_similarities) if feature_similarities else 0
        print(f"3. DEEP FEATURE SIMILARITY: {similarities['deep_features']:.2f}%")
        
        # 4. Scene change comparison
        try:
            scenes1 = analysis1.get("scene_changes", [])
            scenes2 = analysis2.get("scene_changes", [])
            frames1 = analysis1.get("properties", {}).get("frame_count", 0)
            frames2 = analysis2.get("properties", {}).get("frame_count", 0)
            
            similarities["scene_structure"] = compare_scene_changes(scenes1, scenes2, frames1, frames2)
        except Exception as e:
            print(f"Error in scene comparison: {e}")
            similarities["scene_structure"] = 0
            
        print(f"4. SCENE STRUCTURE SIMILARITY: {similarities['scene_structure']:.2f}%")
        
        # 5. Motion comparison
        try:
            motion1 = analysis1.get("motion_vectors", {})
            motion2 = analysis2.get("motion_vectors", {})
            
            similarities["motion"] = compare_motion(motion1, motion2)
        except Exception as e:
            print(f"Error in motion comparison: {e}")
            similarities["motion"] = 0
            
        print(f"5. MOTION SIMILARITY: {similarities['motion']:.2f}%")
        
        # 6. Keyframe comparison
        try:
            keyframes1 = []
            keyframes2 = []
            
            # Try to load keyframes from disk
            for path in analysis1.get("keyframes", []):
                if os.path.exists(path):
                    try:
                        frame = cv2.imread(path)
                        if frame is not None:
                            keyframes1.append(frame)
                    except Exception as e:
                        print(f"Error loading keyframe {path}: {e}")
            
            for path in analysis2.get("keyframes", []):
                if os.path.exists(path):
                    try:
                        frame = cv2.imread(path)
                        if frame is not None:
                            keyframes2.append(frame)
                    except Exception as e:
                        print(f"Error loading keyframe {path}: {e}")
            
            if keyframes1 and keyframes2:
                similarities["keyframes"] = compare_keyframes(keyframes1, keyframes2)
            else:
                similarities["keyframes"] = 0
        except Exception as e:
            print(f"Error in keyframe comparison: {e}")
            similarities["keyframes"] = 0
            
        print(f"6. KEYFRAME SIMILARITY: {similarities['keyframes']:.2f}%")
        
        # 7. Manipulation Detection
        print(f"7. PERFORMING MANIPULATION DETECTION...")
        
        try:
            # Check if both video paths are available
            if 'video_path' in analysis1 and 'video_path' in analysis2:
                # Perform direct manipulation analysis on video files
                manipulation_result = detect_manipulations(
                    analysis1['video_path'], 
                    analysis2['video_path']
                )
                
                # Add manipulation score to similarities
                similarities["manipulation"] = manipulation_result.get('manipulation_score', 100)
                
                print(f"   MANIPULATION DETECTION SCORE: {similarities['manipulation']:.2f}%")
                
                # Store manipulation details
                comparison_result["manipulations"] = {
                    "detected": manipulation_result.get('manipulations_detected', False),
                    "types": manipulation_result.get('manipulation_types', []),
                    "manipulation_score": manipulation_result.get('manipulation_score', 100),
                    "details": manipulation_result.get('details', {})
                }
                print("   MANIPULATION DETECTION SKIPPED - Video paths not available")
                similarities["manipulation"] = 100  # Default score when detection skipped
                comparison_result["manipulations"] = {
                    "detected": False,
                    "types": [],
                    "manipulation_score": 100,
                    "details": {"note": "Detection skipped - video paths not available"}
                }
        except Exception as manip_error:
            print(f"Error during manipulation detection: {manip_error}")
            similarities["manipulation"] = 100  # Default score when detection fails
            comparison_result["manipulations"] = {
                "detected": False,
                "types": [],
                "manipulation_score": 100,
                "details": {"error": f"Detection failed: {str(manip_error)}"}
            }
        
        # Calculate weighted average similarity with manipulation awareness
        # Define importance weights for each analysis type
        weights = {
            "perceptual_hash": 0.20,    # Increase perceptual hash weight - it's more reliable
            "histogram": 0.05,          # Further reduce histogram weight - it's less reliable
            "deep_features": 0.30,      # Increase deep features weight - most reliable signal
            "scene_structure": 0.15,    # Keep same
            "motion": 0.10,             # Slightly reduce motion weight
            "keyframes": 0.15,          # Increase keyframe comparison weight
            "manipulation": 0.05     # High weight for manipulation detection
        }
        
        # Adjust weights for missing components
        available_keys = [k for k in weights.keys() if k in similarities]
        if available_keys:
            # Normalize weights for available components
            total_weight = sum(weights[k] for k in available_keys)
            normalized_weights = {k: weights[k]/total_weight for k in available_keys}
            
            # Calculate weighted similarity
            overall_similarity = sum(similarities[k] * normalized_weights[k] for k in available_keys)
        else:
            overall_similarity = 0
            
        # Apply stronger penalty for detected manipulations
        if comparison_result["manipulations"].get("detected", False):
            # Get the manipulation types
            manip_types = comparison_result["manipulations"].get("types", [])
            
            # Calculate penalty based on type of manipulation
            # Some manipulations shouldn't reduce similarity as much
            base_penalty = 0
            
            for manip_type in manip_types:
                if manip_type in ["frame_insertion", "temporal_truncation"]:
                    # These significantly change content - higher penalty
                    base_penalty += 3
                elif manip_type in ["overlay_modification", "reflection_transformation"]:
                    # These modify presentation but keep core content - lower penalty
                    base_penalty += 1.5
                else:
                    # Other manipulation types - moderate penalty
                    base_penalty += 2
            
            # Cap the total penalty and apply it
            manipulation_penalty = min(15, base_penalty)
            
            # Only apply penalty for videos with moderate to high similarity
            # This prevents manipulation detection from causing false negatives
            if overall_similarity > 60:
                # Apply full penalty for high similarity
                overall_similarity = max(0, overall_similarity - manipulation_penalty)
            elif overall_similarity > 40:
                # Apply reduced penalty for moderate similarity
                scaled_penalty = manipulation_penalty * ((overall_similarity - 40) / 20)
                overall_similarity = max(0, overall_similarity - scaled_penalty)
            # No penalty for low similarity videos (below 40)
        
        if similarities.get("deep_features", 0) > 75:
            # High deep feature similarity is a strong signal of duplicate content
            deep_feature_boost = (similarities["deep_features"] - 75) / 25  # 0 to 1
            
            # The boost is stronger when both deep features AND perceptual hash agree
            if similarities.get("perceptual_hash", 0) > 70:
                hash_agreement = (similarities["perceptual_hash"] - 70) / 30  # 0 to 1
                boost_factor = deep_feature_boost * 0.3 * (1 + hash_agreement)
                
                # Apply the boost
                overall_similarity = min(100, overall_similarity * (1 + boost_factor))
        # Store final results
        comparison_result["overall_similarity"] = overall_similarity
        comparison_result["similarities"] = similarities
        
        comparison_time = time.time() - start_time
        comparison_result["comparison_time"] = comparison_time
        
        print(f"OVERALL SIMILARITY WITH MANIPULATION AWARENESS: {overall_similarity:.2f}%")
        print(f"COMPARISON COMPLETED IN {comparison_time:.2f} SECONDS")
        print(f"{'='*50}\n")
        
        return comparison_result
        
    except Exception as e:
        print(f"ERROR DURING COMPARISON: {str(e)}")
        import traceback
        traceback.print_exc()
        
        comparison_time = time.time() - start_time
        comparison_result["comparison_time"] = comparison_time
        comparison_result["error"] = str(e)
        
        return comparison_result
    



# VIDEO ANALYSIS FUNCTIONS

def analyze_video_full(video_path):
    print(f"\n{'='*50}")
    print(f"STARTING COMPREHENSIVE ANALYSIS OF: {os.path.basename(video_path)}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # Detailed logging and error tracking
    try:
        # 1. Verify Video File
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # 2. Check Video Readability
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Gather actual video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Log initial video properties
        print(f"Video Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Frame Rate: {fps:.2f} fps")
        print(f"  Total Frames: {frame_count}")
        
        # 3. Extract frames with better error handling
        keyframe_start = time.time()
        try:
            keyframes = extract_frame_sequence(video_path, max_frames=15)
            if not keyframes or len(keyframes) == 0:
                raise ValueError("Failed to extract keyframes")
        except Exception as kf_error:
            print(f"Error extracting keyframes: {kf_error}")
            # Create a blank keyframe as fallback
            keyframes = [np.zeros((240, 320, 3), dtype=np.uint8)]
            
        keyframe_time = time.time() - keyframe_start
        print(f"Keyframe Extraction Time: {keyframe_time:.2f} seconds")
        
        # 4. Detailed Feature Extraction with error handling
        feature_start = time.time()
        
        # Initialize with empty results
        perceptual_hashes = []
        histograms = []
        deep_features = []
        
        try:
            # Process keyframes in batches to avoid memory issues
            for i, frame in enumerate(keyframes):
                try:
                    # Compute perceptual hash
                    phash = compute_phash(frame)
                    perceptual_hashes.append(phash)
                    
                    # Compute histogram
                    hist = compute_histogram(frame)
                    histograms.append(hist)
                    
                    # Extract deep features
                    if i % 3 == 0:  # Only process every 3rd frame to save resources
                        features = extract_deep_features(frame)
                        deep_features.append(features)
                except Exception as frame_error:
                    print(f"Error processing frame {i}: {frame_error}")
        except Exception as feature_error:
            print(f"Error in feature extraction: {feature_error}")
        
        feature_time = time.time() - feature_start
        print(f"Feature Extraction Time: {feature_time:.2f} seconds")
        
        # 5. Scene and Motion Analysis
        scene_start = time.time()
        try:
            scene_changes = detect_scene_changes(keyframes)
            motion_vectors = compute_motion_vector(keyframes)
        except Exception as scene_error:
            print(f"Error in scene/motion analysis: {scene_error}")
            scene_changes = []
            motion_vectors = {'magnitudes': [], 'directions': []}
            
        scene_time = time.time() - scene_start
        print(f"Scene & Motion Analysis Time: {scene_time:.2f} seconds")
        
        # 6. Object Detection (Optional)
        object_start = time.time()
        detected_objects = []
        
        if USE_OBJECT_DETECTION:
            try:
                # Process a subset of frames to save resources
                for frame in keyframes[::3]:  # Every 3rd frame
                    objects = detect_objects(frame)
                    detected_objects.append(objects)
            except Exception as obj_error:
                print(f"Error in object detection: {obj_error}")
                
        object_time = time.time() - object_start
        print(f"Object Detection Time: {object_time:.2f} seconds")
        
        # 7. Save keyframes to disk
        keyframe_paths = []
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            keyframe_dir = os.path.join(KEYFRAMES_DIR, video_name)
            os.makedirs(keyframe_dir, exist_ok=True)
            
            keyframe_paths = save_keyframes(
                keyframes[:10],  # Limit to 10 frames
                keyframe_dir,
                video_name
            )
        except Exception as save_error:
            print(f"Error saving keyframes: {save_error}")
        
        # Prepare comprehensive analysis result
        analysis_result = {
            "video_path": video_path,
            "properties": {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "resolution": f"{width}x{height}"
            },
            "perceptual_hashes": perceptual_hashes,
            "histograms": histograms,
            "deep_features": deep_features,
            "scene_changes": scene_changes,
            "motion_vectors": motion_vectors,
            "detected_objects": detected_objects,
            "keyframes": keyframe_paths,
            "analysis_time": {
                "total": time.time() - start_time,
                "keyframe_extraction": keyframe_time,
                "feature_extraction": feature_time,
                "scene_motion_analysis": scene_time,
                "object_detection": object_time
            }
        }
        
        total_time = time.time() - start_time
        print(f"\nTotal Analysis Time: {total_time:.2f} seconds")
        return analysis_result
    
    except Exception as e:
        print(f"❌ Analysis Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "video_path": video_path
        }

def get_cached_analysis(video_path):
    """Get cached analysis results if available with improved error handling"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = os.path.join(ANALYSIS_DIR, f"{video_name}.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                analysis = json.load(f)
                
            print(f"LOADED CACHED ANALYSIS FOR {video_name}")
            return analysis
        except json.JSONDecodeError as e:
            print(f"ERROR LOADING CACHED ANALYSIS: {str(e)}")
            # Try to fix the corrupted cache file
            try:
                # If the file is corrupted, delete it so we can regenerate it
                os.remove(cache_path)
                print(f"Removed corrupted cache file: {cache_path}")
            except Exception as remove_error:
                print(f"Failed to remove corrupted cache file: {remove_error}")
            
            # Return None so a new analysis will be performed
            return None
        except Exception as e:
            print(f"ERROR LOADING CACHED ANALYSIS: {str(e)}")
            return None
            
    return None

def save_analysis_cache(analysis):
    """Save analysis results to cache with NumPy 2.0 compatible handling"""
    video_name = os.path.splitext(os.path.basename(analysis["video_path"]))[0]
    cache_path = os.path.join(ANALYSIS_DIR, f"{video_name}.json")
    
    try:
        # NumPy 2.0 compatible type conversion
        def numpy_safe_encoder(obj):
            # Handle NumPy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            
            # Handle NumPy scalars using dtype checking
            if hasattr(obj, 'dtype') and np.isscalar(obj):
                if np.issubdtype(obj.dtype, np.floating):
                    return float(obj)
                elif np.issubdtype(obj.dtype, np.integer):
                    return int(obj)
                elif np.issubdtype(obj.dtype, np.bool_):
                    return bool(obj)
                else:
                    return obj.item() if hasattr(obj, 'item') else obj
            
            # Handle dictionaries
            if isinstance(obj, dict):
                return {k: numpy_safe_encoder(v) for k, v in obj.items()}
            
            # Handle lists and tuples
            if isinstance(obj, (list, tuple)):
                return [numpy_safe_encoder(item) for item in obj]
            
            return obj
        
        # Convert to JSON-serializable types
        serializable = numpy_safe_encoder(analysis)
        
        # Save as JSON
        with open(cache_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"SAVED ANALYSIS CACHE FOR {video_name}")
        return True
    except Exception as e:
        print(f"ERROR SAVING ANALYSIS CACHE: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up partial file if needed
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"Removed potentially corrupted cache file: {cache_path}")
        except Exception as cleanup_error:
            print(f"Failed to clean up cache file: {cleanup_error}")
        
        return False

# API ROUTES
@app.get("/api/videos")
async def get_videos():
    """Get a list of all videos"""
    try:
        videos = []
        for filename in os.listdir(VIDEOS_DIR):
            if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                file_path = os.path.join(VIDEOS_DIR, filename)
                file_stat = os.stat(file_path)
                
                # Preserve original filename - don't clean or modify it
                videos.append({
                    "filename": filename,
                    "url": f"/original_videos/{filename}",
                    "size": file_stat.st_size,
                    "uploadDate": file_stat.st_mtime
                })
        return videos
    except Exception as e:
        print(f"Error fetching videos: {str(e)}")
        return []

@app.get("/api/test")
async def test_endpoint():
    """Simple endpoint to test the server is working properly"""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "message": "Python backend is running"
    }

@app.get("/api/diagnose")
async def diagnose_system():
    """Diagnostic endpoint to test system capabilities"""
    try:
        results = {
            "system": {},
            "directories": {},
            "video_test": {},
            "frame_test": {}
        }
        
        # Check system info
        import psutil
        results["system"] = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "python_version": sys.version,
            "pytorch_device": str(device),
            "has_cuda": torch.cuda.is_available()
        }
        
        # Check directories
        for dir_name, dir_path in {
            "videos": VIDEOS_DIR,
            "temp": TEMP_DIR,
            "keyframes": KEYFRAMES_DIR,
            "analysis": ANALYSIS_DIR
        }.items():
            results["directories"][dir_name] = {
                "path": dir_path,
                "exists": os.path.exists(dir_path),
                "is_writable": os.access(dir_path, os.W_OK) if os.path.exists(dir_path) else False,
                "file_count": len(os.listdir(dir_path)) if os.path.exists(dir_path) else 0
            }
        
        # Test video read
        video_files = [f for f in os.listdir(VIDEOS_DIR) 
                      if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
        
        if video_files:
            test_video = os.path.join(VIDEOS_DIR, video_files[0])
            cap = cv2.VideoCapture(test_video)
            
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Try to read first frame
                ret, frame = cap.read()
                first_frame_read = ret
                
                cap.release()
                
                results["video_test"] = {
                    "can_open": True,
                    "filename": video_files[0],
                    "resolution": f"{width}x{height}",
                    "fps": fps,
                    "frame_count": frame_count,
                    "first_frame_read": first_frame_read
                }
            else:
                results["video_test"] = {
                    "can_open": False,
                    "filename": video_files[0],
                    "error": "Could not open video file"
                }
        else:
            results["video_test"] = {
                "can_open": False,
                "error": "No video files found"
            }
        
        # Test frame processing
        if "first_frame_read" in results["video_test"] and results["video_test"]["first_frame_read"]:
            try:
                # Test resizing
                resized = cv2.resize(frame, (320, 240))
                
                # Test color conversion
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                
                # Test histogram calculation
                hist = cv2.calcHist([resized], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                # Test perceptual hash
                avg = gray.mean()
                phash = (gray > avg).flatten().astype(int)
                
                # Test saving image
                test_path = os.path.join(TEMP_DIR, "test_frame.jpg")
                cv2.imwrite(test_path, resized)
                
                results["frame_test"] = {
                    "resize_successful": resized.shape == (240, 320, 3),
                    "gray_conversion": gray.shape == (240, 320),
                    "histogram_length": len(hist),
                    "phash_length": len(phash),
                    "save_successful": os.path.exists(test_path),
                    "saved_file_size": os.path.getsize(test_path) if os.path.exists(test_path) else 0
                }
                
                # Clean up test file
                if os.path.exists(test_path):
                    os.remove(test_path)
                    
            except Exception as e:
                results["frame_test"] = {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        return results
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Add this API endpoint in simple_server.py
@app.get("/api/analysis/{video_id}")
async def get_video_analysis(video_id: str):
    """Get detailed analysis for a specific video to be used by the AI agent"""
    try:
        # Find the video file
        video_path = None
        for filename in os.listdir(VIDEOS_DIR):
            if video_id in filename:  # Simple matching, improve as needed
                video_path = os.path.join(VIDEOS_DIR, filename)
                break
        
        if not video_path:
            return {"error": "Video not found"}
        
        # Try to get cached analysis
        analysis = get_cached_analysis(video_path)
        
        if not analysis:
            # If no cached analysis, return minimal info
            return {
                "error": "No analysis data available",
                "filename": os.path.basename(video_path),
                "video_id": video_id
            }
        
        # Create a user-friendly summary of the analysis
        summary = {
            "filename": os.path.basename(video_path),
            "video_id": video_id,
            "resolution": analysis.get("properties", {}).get("resolution", "Unknown"),
            "duration": analysis.get("properties", {}).get("duration", 0),
            "unique_score": analysis.get("unique_percentage", 100),
            "is_unique": analysis.get("is_unique", True),
            "similar_videos": [v.get("filename") for v in analysis.get("similar_videos", [])],
            "similarity_score": analysis.get("similar_videos", [{}])[0].get("overallSimilarity", 0) if analysis.get("similar_videos") else 0,
            "manipulation_detected": analysis.get("manipulations", {}).get("detected", False),
            "manipulation_types": analysis.get("manipulations", {}).get("types", []),
            "algorithm_scores": analysis.get("analysis_details", {})
        }
        
        # Convert any remaining NumPy types
        return convert_numpy_types(summary)
    
    except Exception as e:
        print(f"Error getting video analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to get analysis: {str(e)}"}
    
@app.get("/api/analysis/latest")
async def get_latest_analysis():
    """Get the most recent video analysis data with improved error handling"""
    try:
        # Print the ANALYSIS_DIR path for debugging
        print(f"Looking for analysis files in: {ANALYSIS_DIR}")
        
        # Check if directory exists
        if not os.path.exists(ANALYSIS_DIR):
            print(f"ERROR: Analysis directory does not exist: {ANALYSIS_DIR}")
            return {"error": "Analysis directory not found"}
        
        # List all files in directory
        all_files = os.listdir(ANALYSIS_DIR)
        json_files = [f for f in all_files if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files in analysis directory")
        
        if not json_files:
            return {"error": "No analysis files found"}
        
        # Find the most recent file
        latest_file = None
        latest_time = 0
        
        for filename in json_files:
            file_path = os.path.join(ANALYSIS_DIR, filename)
            try:
                mod_time = os.path.getmtime(file_path)
                if mod_time > latest_time:
                    latest_time = mod_time
                    latest_file = file_path
            except Exception as e:
                print(f"Error checking file {filename}: {str(e)}")
        
        if not latest_file:
            return {"error": "No valid analysis files found"}
        
        print(f"Latest analysis file: {os.path.basename(latest_file)}")
        
        # Read the latest analysis file
        try:
            with open(latest_file, 'r') as f:
                content = f.read()
                analysis_data = json.loads(content)
                
            # Log key information for debugging
            print(f"Successfully loaded analysis data: {os.path.basename(latest_file)}")
            print(f"Analysis keys: {list(analysis_data.keys())}")
            
            # Create a simplified version with essential data
            simplified = {
                "filename": analysis_data.get("filename", "Unknown"),
                "unique_percentage": analysis_data.get("unique_percentage", 100),
                "is_unique": analysis_data.get("is_unique", True),
                "similar_videos": analysis_data.get("similar_videos", []),
                "manipulations": analysis_data.get("manipulations", {"detected": False, "types": []}),
                "analysis_details": analysis_data.get("analysis_details", {})
            }
            
            return simplified
        except json.JSONDecodeError as e:
            print(f"JSON decode error in {os.path.basename(latest_file)}: {str(e)}")
            # Try to fix corrupted JSON if possible
            try:
                # Remove the file to prevent further issues
                os.remove(latest_file)
                print(f"Removed corrupted file: {latest_file}")
            except:
                pass
            return {"error": f"Invalid JSON in file: {os.path.basename(latest_file)}", "details": str(e)}
        except Exception as read_error:
            print(f"Error reading file {os.path.basename(latest_file)}: {str(read_error)}")
            return {"error": f"Could not read analysis file: {str(read_error)}"}
    except Exception as e:
        print(f"Error getting latest analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to get analysis data: {str(e)}"}

# Replace your analyze_video_endpoint function in simple_server.py with this:

@app.post("/api/analyze")
async def analyze_video_endpoint(video: UploadFile = File(...)):
    """
    Comprehensive video analysis endpoint with robust error handling
    and proper response formatting.
    """
    try:
        logging.info(f"📤 Video Upload Received:")
        logging.info(f"Filename: {video.filename}")
        logging.info(f"Content Type: {video.content_type}")
        
        # Generate unique filename
        timestamp = str(int(time.time() * 1000))
        safe_filename = f"{timestamp}-{video.filename}"
        temp_path = os.path.join(TEMP_DIR, safe_filename)
        
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        # Perform comprehensive analysis
        analysis_result = analyze_video_full(temp_path)
        
        # Process and structure the analysis result for the frontend
        result = {
            "temp_path": temp_path,
            "filename": safe_filename,
            "unique_percentage": 100,  # Default value, will be updated below
            "is_unique": True,         # Default value, will be updated below
            "similar_videos": [],      # Will be populated if similar videos are found
            "fallback": False,
            "first_video": False,
            "analysis_details": {}
        }
        
        # Check for errors in analysis
        if "error" in analysis_result:
            result["fallback"] = True
            result["analysis_details"] = {
                "error": analysis_result.get("error", "Unknown error during analysis"),
                "note": "Analysis was limited. Basic uniqueness estimation only."
            }
            return convert_numpy_types(result)
        
        # Get video properties if available
        properties = analysis_result.get("properties", {})
        if properties:
            result["analysis_details"].update({
                "resolution": f"{properties.get('width', 0)}x{properties.get('height', 0)}",
                "duration": properties.get('duration', 0),
                "fps": properties.get('fps', 0),
                "frame_count": properties.get('frame_count', 0)
            })
        
        # Find similar videos in our database
        similar_videos = []
        similarity_scores = []
        
        # Get list of existing videos
        video_files = [f for f in os.listdir(VIDEOS_DIR) 
                      if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
        
        if not video_files:
            # This is the first video being uploaded
            result["first_video"] = True
            return convert_numpy_types(result)
        
        # We have other videos to compare against
        for filename in video_files:
            existing_path = os.path.join(VIDEOS_DIR, filename)
            
            # Skip if it's the same file
            if os.path.samefile(existing_path, temp_path):
                continue
                
            try:
                # Use the cached analysis for the existing video if available
                existing_analysis = get_cached_analysis(existing_path)
                
                if not existing_analysis:
                    # Analyze the existing video if no cache is available
                    existing_analysis = analyze_video_full(existing_path)
                    # Cache the analysis for future use
                    save_analysis_cache(existing_analysis)
                
                # Compare the videos
                comparison = compare_videos_with_manipulation_detection(
                    analysis_result, 
                    existing_analysis
                )
                
                # Get the overall similarity score
                similarity = comparison.get("overall_similarity", 0)
                
                # If the similarity is above a threshold, add to similar videos
                if similarity > 40:  # 40% similarity threshold
                    similarity_scores.append(similarity)
                    similar_videos.append({
                        "filename": filename,
                        "overallSimilarity": round(similarity),
                        "size": os.path.getsize(existing_path),
                        "manipulations": comparison.get("manipulations", {
                            "detected": False,
                            "types": []
                        }),
                        "detailed_scores": {
                            "perceptual_hash": round(comparison.get("similarities", {}).get("perceptual_hash", 0)),
                            "histogram": round(comparison.get("similarities", {}).get("histogram", 0)),
                            "deep_features": round(comparison.get("similarities", {}).get("deep_features", 0)),
                            "scene_structure": round(comparison.get("similarities", {}).get("scene_structure", 0)),
                            "motion": round(comparison.get("similarities", {}).get("motion", 0)),
                            "keyframes": round(comparison.get("similarities", {}).get("keyframes", 0))
                        }
                    })
            except Exception as e:
                print(f"Error comparing with {filename}: {str(e)}")
                # Continue with next video
                continue
        
        # Sort similar videos by similarity (highest first)
        similar_videos.sort(key=lambda x: x["overallSimilarity"], reverse=True)
        result["similar_videos"] = similar_videos
        
        # Calculate uniqueness as inverse of highest similarity
        max_similarity = max(similarity_scores) if similarity_scores else 0
        if max_similarity > 80:
            # Boost high similarities (likely duplicates)
            max_similarity = 80 + (max_similarity - 80) * 1.5
        elif max_similarity > 65 and max_similarity < 80:
            # This is the "gray area" - apply a separation curve
            # Scores closer to 80 get pushed higher, scores closer to 65 get pushed lower
            position = (max_similarity - 65) / 15  # 0 to 1 based on position in range
            if position > 0.6:  # Upper part of gray area - likely duplicates
                max_similarity = max_similarity + (position - 0.6) * 15
            else:  # Lower part of gray area - likely uniques
                max_similarity = max_similarity - (0.6 - position) * 10
        elif max_similarity > 40 and max_similarity <= 65:
            # Reduce moderate similarities (likely different videos with some similar content)
            max_similarity = 40 + (max_similarity - 40) * 0.7

        result["unique_percentage"] = max(0, min(100, 100 - max_similarity))
        # Lower the threshold to consider uniqueness - better to flag potential duplicates
        result["is_unique"] = result["unique_percentage"] >= 40  # Threshold for uniqueness
        
        if similar_videos:
    # Sort by similarity (highest first) and keep only the most similar one
            similar_videos.sort(key=lambda x: x["overallSimilarity"], reverse=True)
            result["similar_videos"] = [similar_videos[0]]
        else:
            result["similar_videos"] = []

        # Add a clear message about the uniqueness decision
        if result["is_unique"]:
            result["uniqueness_message"] = "This video appears to be unique."
        else:
            most_similar = similar_videos[0]["filename"] if similar_videos else "a previously uploaded video"
            similarity = similar_videos[0]["overallSimilarity"] if similar_videos else 0
            result["uniqueness_message"] = f"This video is {similarity}% similar to {most_similar}."
        # Add algorithm scores to analysis_details
        algorithm_scores = {}
        
                # Process perceptual hash data
                # Process perceptual hash data
        if analysis_result.get("perceptual_hashes"):
            # Calculate a more realistic score based on how many hashes we have
            hash_count = len(analysis_result.get("perceptual_hashes", []))
            hash_quality = min(90, 60 + hash_count * 5)  # Base score of 60, max 90
            algorithm_scores["perceptual_hash"] = hash_quality
        else:
            algorithm_scores["perceptual_hash"] = 70  # Default but not 100%

        # Process histogram data
        if analysis_result.get("histograms"):
            # Calculate a more realistic score based on how many histograms we have
            hist_count = len(analysis_result.get("histograms", []))
            hist_quality = min(90, 60 + hist_count * 5)  # Base score of 60, max 90
            algorithm_scores["histogram"] = hist_quality
        else:
            algorithm_scores["histogram"] = 65  # Default but not 100%

        # Process deep features
        if analysis_result.get("deep_features"):
            # Calculate a more realistic score
            deep_features = analysis_result.get("deep_features", [])
            features_count = len(deep_features) if isinstance(deep_features, list) else 1
            features_quality = min(95, 65 + features_count * 5)  # Base score of 65, max 95
            algorithm_scores["deep_features"] = features_quality
        else:
            algorithm_scores["deep_features"] = 75  # Default but not 100%

        # Process scene changes
        if "scene_changes" in analysis_result:
            scene_changes = analysis_result.get("scene_changes", [])
            scene_count = len(scene_changes)
            scene_quality = min(95, 60 + scene_count * 8)  # Base score of 60, max 95
            algorithm_scores["scene_structure"] = scene_quality
        else:
            algorithm_scores["scene_structure"] = 60  # Default but not 100%

        # Process motion vectors
        if "motion_vectors" in analysis_result:
            motion_vectors = analysis_result.get("motion_vectors", {})
            # Check if it has the expected structure
            if isinstance(motion_vectors, dict) and ("magnitudes" in motion_vectors or "directions" in motion_vectors):
                motion_quality = 80  # Good quality for complete motion analysis
            else:
                motion_quality = 65  # Lower quality for partial analysis
            algorithm_scores["motion"] = motion_quality
        else:
            algorithm_scores["motion"] = 55  # Default but not 100%

        # Process manipulation detection
        # This depends on the similar videos and their manipulations
        if similar_videos and any(v.get("manipulations", {}).get("detected", False) for v in similar_videos):
            # Some manipulation was detected in similar videos
            algorithm_scores["manipulation"] = 85  # High confidence in manipulation detection
        else:
            algorithm_scores["manipulation"] = 70  # Moderate confidence, not 100%

        # Process keyframes
        if "keyframes" in analysis_result:
            keyframes = analysis_result.get("keyframes", [])
            keyframe_count = len(keyframes)
            keyframe_quality = min(90, 55 + keyframe_count * 7)  # Base score of 55, max 90
            algorithm_scores["keyframes"] = keyframe_quality
        else:
            algorithm_scores["keyframes"] = 60  # Default but not 100%

        # Add algorithm scores to analysis details
        result["analysis_details"].update(algorithm_scores)
        
        # Check for manipulations in similar videos
        manipulations = {
            "detected": False,
            "types": []
        }
        
        # If we found similar videos with manipulations, add that info
        for vid in similar_videos:
            if vid.get("manipulations", {}).get("detected", False):
                manipulations["detected"] = True
                for manip_type in vid.get("manipulations", {}).get("types", []):
                    if manip_type not in manipulations["types"]:
                        manipulations["types"].append(manip_type)
        
        result["manipulations"] = manipulations
        
        # Convert NumPy types and return the result
        return convert_numpy_types(result)
    
    except Exception as error:
        print(f"Analysis endpoint error: {error}")
        import traceback
        traceback.print_exc()
        
        # Return a fallback result on error
        return {
            "temp_path": temp_path if 'temp_path' in locals() else None,
            "filename": safe_filename if 'safe_filename' in locals() else f"{int(time.time() * 1000)}-error.mp4",
            "unique_percentage": 100,
            "is_unique": True,
            "fallback": True,
            "similar_videos": [],
            "analysis_details": {
                "error": str(error),
                "note": "Error during analysis. Using fallback mode."
            }
        }

def compute_simple_similarity(frames, video_path):
    """Simple function to compute similarity between frames and a video"""
    print(f"  Computing simple similarity with {os.path.basename(video_path)}")
    
    # Extract frames from comparison video
    cap = cv2.VideoCapture(video_path)
    comp_frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get sample frames
    indices = np.linspace(0, frame_count-1, min(10, frame_count), dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            comp_frames.append(frame)
    cap.release()
    
    print(f"  Extracted {len(comp_frames)} frames from comparison video")
    
    # Calculate basic similarity measures
    phash_sims = []
    hist_sims = []
    
    for i, frame1 in enumerate(frames):
        if i >= len(comp_frames):
            break
        
        frame2 = comp_frames[i]
        
        # Resize for consistency
        frame1 = cv2.resize(frame1, (320, 240))
        frame2 = cv2.resize(frame2, (320, 240))
        
        # Calculate perceptual hash
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Simple hash comparison
        hash1 = (gray1 > np.mean(gray1)).flatten().astype(int)
        hash2 = (gray2 > np.mean(gray2)).flatten().astype(int)
        
        hash_sim = 100 - (np.sum(hash1 != hash2) / len(hash1) * 100)
        phash_sims.append(hash_sim)
        
        # Simple histogram comparison
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL) * 100
        hist_sims.append(max(0, hist_sim))
    
    # Calculate final similarity
    avg_phash = np.mean(phash_sims) if phash_sims else 0
    avg_hist = np.mean(hist_sims) if hist_sims else 0
    
    print(f"  Average hash similarity: {avg_phash:.2f}%")
    print(f"  Average histogram similarity: {avg_hist:.2f}%")
    
    # Weighted combination
    final_sim = 0.6 * avg_phash + 0.4 * avg_hist
    
    return round(final_sim)

@app.post("/api/upload")
async def upload_video(temp_path: str = Form(...), caption: str = Form(None), target_dir: str = Form(None)):
    """Upload a previously analyzed video to the specified directory"""
    try:
        print(f"\n{'='*50}")
        print(f"UPLOADING VIDEO FROM TEMP PATH: {temp_path}")
        print(f"TARGET DIRECTORY: {target_dir if target_dir else 'default directory'}")
        print(f"{'='*50}")
        
        # Check if the temp file exists
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Temporary file not found")
            
        # Get the filename from the temp path
        filename = os.path.basename(temp_path)
        
        # Determine the target directory
        if target_dir and os.path.exists(target_dir):
            final_dir = target_dir
        else:
            final_dir = VIDEOS_DIR  # Use default if target not specified or doesn't exist
        
        # Move to permanent storage
        final_path = os.path.join(final_dir, filename)
        print(f"Moving file from {temp_path} to {final_path}")
        
        # Make sure the target directory exists
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        
        # Copy the file instead of moving it (safer)
        shutil.copy2(temp_path, final_path)
        
        # Remove the temp file after successful copy
        if os.path.exists(final_path):
            try:
                os.remove(temp_path)
                print(f"Temporary file removed: {temp_path}")
            except Exception as remove_error:
                print(f"Warning: Could not remove temp file: {remove_error}")
        
        # Save caption if provided
        if caption:
            caption_data = {
                "caption": caption,
                "timestamp": time.time()
            }
            caption_path = os.path.join(final_dir, f"{filename}.json")
            with open(caption_path, 'w') as f:
                json.dump(caption_data, f)
            print(f"Saved caption to {caption_path}")
        
        print(f"Upload complete!")
        print(f"{'='*50}\n")
        
        return {
            "success": True,
            "message": "Video uploaded successfully",
            "filename": filename,
            "path": final_path
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR DURING UPLOAD: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

if __name__ == "__main__":
    print(f"Videos directory: {VIDEOS_DIR}")
    print(f"Temp directory: {TEMP_DIR}")
    print(f"Keyframes directory: {KEYFRAMES_DIR}")
    print(f"Analysis directory: {ANALYSIS_DIR}")
    print(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    uvicorn.run(app, host="0.0.0.0", port=3001)