from flask import Flask, request, jsonify
from flask_cors import CORS
from gpt4all import GPT4All
import os
import json
import glob
import traceback

class ProjectFileExplorer:
    def __init__(self, project_root):
        self.project_root = project_root

    def list_videos(self, directory=None):
        """
        List videos in a specified directory with robust error handling
        """
        if not directory:
            directory = os.path.join(
                self.project_root, 
                'echo-hall', 'server', 'public', 'original_videos'
            )
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        try:
            video_files = []
            for f in os.listdir(directory):
                full_path = os.path.join(directory, f)
                if os.path.isfile(full_path) and os.path.splitext(f)[1].lower() in video_extensions:
                    stats = os.stat(full_path)
                    video_files.append({
                        'filename': f,
                        'path': full_path,
                        'size': stats.st_size,
                        'modified': stats.st_mtime
                    })
            
            # Sort videos by modification time, most recent first
            video_files.sort(key=lambda x: x['modified'], reverse=True)
            
            return {
                'total_videos': len(video_files),
                'videos': video_files
            }
        except Exception as e:
            return {
                'error': str(e),
                'directory': directory
            }

    def list_keyframes(self, directory=None):
        """
        List and analyze keyframes with detailed information
        """
        if not directory:
            directory = os.path.join(
                self.project_root, 
                'echo-hall', 'server', 'public', 'keyframes'
            )
        
        try:
            keyframe_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            keyframe_files = []
            
            for root, _, files in os.walk(directory):
                for file in files:
                    if os.path.splitext(file)[1].lower() in keyframe_extensions:
                        full_path = os.path.join(root, file)
                        stats = os.stat(full_path)
                        
                        # Extract video name from keyframe filename
                        try:
                            # Assume filename format like 'videoname_frame_001.jpg'
                            parts = file.split('_')
                            video_name = parts[0] if len(parts) > 1 else 'Unknown'
                        except:
                            video_name = 'Unknown'
                        
                        keyframe_files.append({
                            'filename': file,
                            'video_source': video_name,
                            'full_path': full_path,
                            'relative_path': os.path.relpath(full_path, directory),
                            'size': stats.st_size,
                            'modified': stats.st_mtime
                        })
            
            # Group keyframes by video source
            grouped_keyframes = {}
            for kf in keyframe_files:
                source = kf['video_source']
                if source not in grouped_keyframes:
                    grouped_keyframes[source] = []
                grouped_keyframes[source].append(kf)
            
            return {
                'total_keyframes': len(keyframe_files),
                'keyframes_by_video': grouped_keyframes,
                'summary': {
                    'videos_with_keyframes': len(grouped_keyframes),
                    'keyframes_per_video': {
                        video: len(frames) for video, frames in grouped_keyframes.items()
                    }
                }
            }
        except Exception as e:
            return {
                'error': str(e),
                'directory': directory
            }

class ProjectAIAssistant:
    def __init__(self, project_root, model_name='mistral-7b-instruct-v0.1.Q4_0.gguf'):
        """
        Initialize AI assistant with reduced context size
        """
        self.model = GPT4All(model_name)
        self.file_explorer = ProjectFileExplorer(project_root)

    def generate_response(self, query):
        """
        Generate a concise, context-aware response
        """
        query_lower = query.lower()
        
        try:
            # Specific query handling
            if 'videos' in query_lower:
                video_info = self.file_explorer.list_videos()
                context = f"Video Catalog:\n"
                context += f"Total Videos: {video_info.get('total_videos', 0)}\n"
                context += "Recent Videos:\n"
                for video in video_info.get('videos', [])[:3]:
                    context += f"- {video['filename']} (Size: {video['size']} bytes)\n"
            
            elif 'keyframe' in query_lower:
                keyframe_info = self.file_explorer.list_keyframes()
                context = f"Keyframe Analysis:\n"
                context += f"Total Keyframes: {keyframe_info.get('total_keyframes', 0)}\n"
                context += f"Videos with Keyframes: {keyframe_info.get('summary', {}).get('videos_with_keyframes', 0)}\n"
                context += "Keyframe Distribution:\n"
                for video, count in keyframe_info.get('summary', {}).get('keyframes_per_video', {}).items():
                    context += f"- {video}: {count} keyframes\n"
            
            else:
                context = "Project Overview:\n- Video analysis project\n- Advanced manipulation detection\n- Keyframe and video tracking capabilities"
            
            # Prepare prompt with limited context
            full_prompt = f"""You are an AI assistant for the Echo Hall Video Analysis Project.
Respond concisely and precisely to the following query.

Project Context:
{context}

User Query: {query}

Provide a clear, informative response:"""
            
            # Generate response with reduced tokens
            with self.model.chat_session():
                response = self.model.generate(
                    full_prompt, 
                    max_tokens=300,  # Reduced token count
                    temp=0.5  # More focused response
                )
            
            return {
                'status': 'success',
                'response': response
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'response': f"Error processing query: {str(e)}",
                'trace': traceback.format_exc()
            }

# Flask Application Setup
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize AI Assistant with project root
PROJECT_ROOT = r'C:\Users\dbmkr\Documents\AME 598- AI Social Good\Assignment - Presentation Proposal\echo-hall'
ai_assistant = ProjectAIAssistant(PROJECT_ROOT)

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """
    Handle incoming chat requests with improved error handling
    """
    data = request.json
    
    # Validate input
    if not data or 'message' not in data:
        return jsonify({
            'status': 'error',
            'response': 'No message provided'
        }), 400
    
    # Process message with context awareness
    result = ai_assistant.generate_response(data['message'])
    
    return jsonify(result)

@app.route('/api/explore/videos', methods=['GET'])
def list_videos():
    """
    List videos in the project directory
    """
    videos = ai_assistant.file_explorer.list_videos()
    return jsonify(videos)

@app.route('/api/explore/keyframes', methods=['GET'])
def list_keyframes():
    """
    List keyframes in the project directory
    """
    keyframes = ai_assistant.file_explorer.list_keyframes()
    return jsonify(keyframes)

@app.route('/api/status', methods=['GET'])
def check_status():
    """
    Check AI model availability
    """
    return jsonify({
        'status': 'connected',
        'models': ['mistral-7b-instruct'],
        'project_root': PROJECT_ROOT
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3004, debug=True)  