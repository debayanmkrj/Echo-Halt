let latestAnalysisResult = null;
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');
const axios = require('axios');
const FormData = require('form-data');

const GPT4AllClient = require('./gpt4all-nodejs-client');
const ProjectContextProvider = require('./project-context');

// Initialize Ollama client


const projectRoot = 'C:/Users/dbmkr/Documents/AME 598- AI Social Good/Assignment - Presentation Proposal/echo-hall';
const contextProvider = new ProjectContextProvider(projectRoot);
const ollamaClient = new GPT4AllClient('http://localhost:3004');
const latestAnalysisPath = path.join(__dirname, 'latest_analysis.json');
// Import video analysis utilities
const {
    perceptualHash,
    histogramDiff,
    shotBoundaryDetection,
    extractKeyframes,
    objectDetection,
    visualEmbedding,
    splitScreenDetection,
    chromaKeyDetection
} = require('C:/Users/dbmkr/Documents/AME 598- AI Social Good/Assignment - Presentation Proposal/echo-hall/video_analysis_utils.js');

const app = express();
const PORT = 3002;

const videoAnalysisRouter = express.Router();
const llamaRouter = express.Router();


llamaRouter.use(cors({
    origin: ['http://localhost:3003'],
    credentials: true,
    methods: ['GET', 'POST', 'OPTIONS']
}));

function saveLatestAnalysis(analysisData) {
  try {
    fs.writeFileSync(latestAnalysisPath, JSON.stringify(analysisData, null, 2));
    console.log('Latest analysis saved successfully');
    return true;
  } catch (error) {
    console.error('Failed to save latest analysis:', error.message);
    return false;
  }
}

function getLatestAnalysis() {
  return latestAnalysisResult;
}

llamaRouter.get('/status', async (req, res) => {
    try {
      const isAvailable = await ollamaClient.isAvailable();
      const models = isAvailable ? await ollamaClient.listModels() : [];
      
      res.json({
        status: isAvailable ? 'connected' : 'disconnected',
        models: models.map(model => model.name)
      });
    } catch (error) {
      console.error('Error checking Llama status:', error);
      res.status(500).json({ 
        status: 'error',
        message: 'Failed to check Llama 2 status'
      });
    }
  });
  
  // Route to handle chat messages
  // Route to handle chat messages
// Enhance the route to handle chat messages in server.js
// In server.js, replace or update the llamaRouter.post('/chat') function
// In server.js - Update the llamaRouter.post('/chat') function
// In server.js - update the llamaRouter.post('/chat') function
llamaRouter.post('/chat', async (req, res) => {
  const { message } = req.body;
  
  if (!message || typeof message !== 'string') {
    return res.status(400).json({ 
      error: 'Invalid request. Message is required.'
    });
  }
  
  try {
    // Check if query is relevant to the project
    if (!contextProvider.isRelevantQuery(message)) {
      return res.json({
        response: "I'm focused on answering questions about the Echo Hall video analysis project. Please ask something related to the project files or video analysis capabilities."
      });
    }
    
    // Get project context
    const projectContext = await contextProvider.getProjectContext();
    
    // Use the global analysis data variable
    const analysisData = latestAnalysisResult;
    console.log("Using analysis data:", analysisData ? "Available" : "Not available");
    
    // Create a detailed prompt with analysis data when available
    let prompt = `You are an AI assistant integrated into the Echo Hall video analysis project. 
Your task is to explain video analysis results and project capabilities.

Project context:
${projectContext.substring(0, 500)}... [abbreviated]

`;

    // Add analysis data if available
    if (analysisData) {
      prompt += `
RECENT VIDEO ANALYSIS RESULTS:
=============================
Filename: ${analysisData.filename || "Unknown"}
Uniqueness Score: ${analysisData.unique_percentage || 0}%
Is Unique: ${analysisData.is_unique ? "Yes" : "No"}

`;
      
      // Add similar videos information
      if (analysisData.similar_videos && analysisData.similar_videos.length > 0) {
        const similarVideo = analysisData.similar_videos[0];
        prompt += `
SIMILARITY INFORMATION:
Similar to: ${similarVideo.filename || "Unknown"} 
Similarity Score: ${similarVideo.overallSimilarity || 0}%

Detailed Similarity Scores:
- Perceptual Hash: ${similarVideo.detailed_scores?.perceptual_hash || 0}%
- Histogram: ${similarVideo.detailed_scores?.histogram || 0}%
- Deep Features: ${similarVideo.detailed_scores?.deep_features || 0}%
- Scene Structure: ${similarVideo.detailed_scores?.scene_structure || 0}%
- Motion: ${similarVideo.detailed_scores?.motion || 0}%
- Keyframes: ${similarVideo.detailed_scores?.keyframes || 0}%
`;
      } else {
        prompt += "No similar videos were found in the database.\n";
      }
      
      // Add manipulation detection information
      prompt += `
MANIPULATION DETECTION:
Manipulations Detected: ${analysisData.manipulations?.detected ? "Yes" : "No"}
`;
      
      if (analysisData.manipulations?.detected && analysisData.manipulations.types?.length > 0) {
        prompt += `Types of Manipulations: ${analysisData.manipulations.types.join(", ")}\n`;
      }
      
      // Add algorithm confidence scores
      if (analysisData.analysis_details) {
        prompt += `
ALGORITHM CONFIDENCE SCORES:
`;
        for (const [key, value] of Object.entries(analysisData.analysis_details)) {
          if (typeof value === 'number') {
            prompt += `- ${key}: ${value}\n`;
          }
        }
      }
      
      prompt += `
EXPLANATION OF SCORES:
- Uniqueness Score: Lower percentage means the video is more similar to existing videos. 0% means identical, 100% means completely unique.
- Similarity Score: Higher percentage means more similar to existing videos. 100% means identical, 0% means completely different.
- Perceptual Hash: Measures visual similarity based on image characteristics.
- Histogram: Compares color distribution between videos.
- Deep Features: Uses AI to identify high-level content similarities.
- Scene Structure: Compares how scenes are arranged and transition.
- Motion: Analyzes movement patterns within the videos.
- Keyframes: Compares important frames between videos.
`;
    } else {
      prompt += "\nIMPORTANT: No video analysis data is currently available. You should clearly state this to the user.\n";
    }
    
    // Add user message and instructions
    prompt += `
User message: ${message}

IMPORTANT INSTRUCTIONS:
1. If analysis data is provided above, focus on explaining those specific results.
2. If the analysis shows low uniqueness (under 50%), explain that the video is very similar to existing content.
3. If the analysis shows high uniqueness (over 50%), explain that the video appears to be unique.
4. Mention specific scores from the analysis data when available.
5. If no analysis data is provided above, clearly state that you don't have access to any analysis results.
6. DO NOT make up or hallucinate analysis results.
7. Provide your response in a clear, educational manner.
`;
    
    // Generate response from Llama 2
    const response = await ollamaClient.generateCompletion(prompt, 'llama2');
    
    // Send response to client
    res.json({ response });
  } catch (error) {
    console.error('Error generating Llama 2 response:', error);
    res.status(500).json({
      error: 'Failed to generate response',
      message: error.message
    });
  }
});

  
  

videoAnalysisRouter.use(cors({
    origin: ['http://localhost:3000', 'http://localhost:3002', 'http://localhost:3003', 'http://localhost:3004'],
    credentials: true
}));

// Initialize Express middleware
app.use(cors({
    origin: [
        'http://localhost:3000', 
        'http://localhost:3001', 
        'http://localhost:3002', 
        'http://localhost:3003',
        'http://127.0.0.1:3000',
        'http://127.0.0.1:3001',
        'http://127.0.0.1:3002',
        'http://127.0.0.1:3003'
    ],
    credentials: true,
    methods: ['GET', 'POST', 'OPTIONS']
}));

app.use(express.json());
app.use('/videos', express.static(path.join(__dirname, 'public', 'original_videos')));
app.use('/temp', express.static(path.join(__dirname, 'public', 'temp')));
app.use('/public', express.static(path.join(__dirname, 'public')));
app.use('/original_videos', express.static(path.join(__dirname, 'public', 'original_videos')));
app.use('/keyframes', express.static(path.join(__dirname, 'public', 'keyframes')));





app.get('/api/videos', async (req, res) => {
    try {
        const formData = new FormData();
        console.log("Fetching videos from Python backend...");
        // Get videos from Python backend
        const pythonResponse = await axios.post(`${PYTHON_BACKEND_URL}/api/analyze`, formData, {
            headers: {
              ...formData.getHeaders(),
              'Accept': 'application/json'
            },
            timeout: 300000, // 5-minute timeout
            maxContentLength: Infinity,
            maxBodyLength: Infinity
          });
        
        // Check if the response contains videos
        if (pythonResponse.data && Array.isArray(pythonResponse.data)) {
            console.log(`Found ${pythonResponse.data.length} videos from Python backend`);
            // Make sure the URLs are correctly formatted
            const videos = pythonResponse.data.map(video => {
                // Check if URL is properly formatted
                if (!video.url.startsWith('/')) {
                    video.url = `/${video.url}`;
                }
                // Ensure filename is properly set
                if (!video.filename && video.url) {
                    video.filename = path.basename(video.url);
                }
                return video;
            });
            return res.json(videos);
        } else {
            throw new Error("Invalid response format from Python backend");
        }
    } catch (error) {
        console.error('Error fetching videos from Python backend:', error);
        
        // Fallback to Node.js implementation
        try {
            console.log("Falling back to Node.js implementation for video list");
            const videoFiles = fs.readdirSync(directories.videos)
                .filter(file => file.match(/\.(mp4|mov|avi|mkv)$/i))
                .map(filename => {
                    const filePath = path.join(directories.videos, filename);
                    const stats = fs.statSync(filePath);
                    return {
                        filename: filename,
                        url: `/original_videos/${filename}`,
                        size: stats.size,
                        uploadDate: stats.mtime
                    };
                });

            console.log(`Found ${videoFiles.length} videos from Node.js implementation`);
            res.json(videoFiles);
        } catch (fallbackError) {
            console.error('Error in fallback video fetching:', fallbackError);
            res.status(500).json({
                error: 'Failed to fetch videos',
                details: fallbackError.message
            });
        }
    }
});

// In server.js add this route
// Add this endpoint after the other app.get endpoints
app.get('/api/latest-analysis', (req, res) => {
  try {
    const analysis = getLatestAnalysis();
    if (analysis) {
      res.json(analysis);
    } else {
      res.status(404).json({ error: 'No analysis data found' });
    }
  } catch (error) {
    console.error('Error retrieving latest analysis:', error);
    res.status(500).json({ error: 'Failed to retrieve analysis data' });
  }
});

// Define and create directory structure
const directories = {
    videos: path.join(__dirname, 'public', 'original_videos'),
    temp: path.join(__dirname, 'public', 'temp'),
    analysis: path.join(__dirname, 'public', 'analysis_results')
};

// Create directories and verify permissions
const initializeDirectories = () => {
    Object.entries(directories).forEach(([name, dir]) => {
        try {
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
            // Verify write permissions
            const testFile = path.join(dir, '.test');
            fs.writeFileSync(testFile, '');
            fs.unlinkSync(testFile);
            console.log(`Directory ${name} is ready at: ${dir}`);
        } catch (error) {
            console.error(`Failed to initialize directory ${name}:`, error);
            throw error;
        }
    });
};

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function(req, file, cb) {
        const tempDir = path.join(__dirname, '..', 'public', 'temp');
        cb(null, tempDir);
    },
    filename: function(req, file, cb) {
        const uniqueName = `${Date.now()}-${file.originalname.replace(/\s+/g, '-')}`;
        cb(null, uniqueName);
    }
});

const upload = multer({
    storage: storage,
    limits: {
        fileSize: 200 * 1024 * 1024
    }
});

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:3001';

// Serve static files from our directories
Object.entries(directories).forEach(([key, dir]) => {
    app.use(`/${key}`, express.static(dir));
});

// Helper function to calculate video similarity
async function calculateVideoSimilarity(uploadedVideo, existingVideo) {
    // Analyze both videos
    const uploadedAnalysis = {
        hash: await perceptualHash(uploadedVideo),
        histogram: await histogramDiff(uploadedVideo),
        objects: await objectDetection(uploadedVideo),
        keyframes: await extractKeyframes(uploadedVideo),
        visualEmbedding: await visualEmbedding(uploadedVideo),
        shotBoundaries: await shotBoundaryDetection(uploadedVideo)
    };

    const existingAnalysis = {
        hash: await perceptualHash(existingVideo),
        histogram: await histogramDiff(existingVideo),
        objects: await objectDetection(existingVideo),
        keyframes: await extractKeyframes(existingVideo),
        visualEmbedding: await visualEmbedding(existingVideo),
        shotBoundaries: await shotBoundaryDetection(existingVideo)
    };

    // Define importance weights for each analysis type
    const weights = {
        perceptualHash: 25,
        histogram: 20,
        objects: 20,
        keyframes: 15,
        visualEmbedding: 15,
        shotBoundaries: 5
    };

    // Calculate similarities for each feature
    const similarities = {
        perceptualHash: calculateHashSimilarity(uploadedAnalysis.hash, existingAnalysis.hash),
        histogram: calculateHistogramSimilarity(uploadedAnalysis.histogram, existingAnalysis.histogram),
        objects: calculateObjectSimilarity(uploadedAnalysis.objects, existingAnalysis.objects),
        keyframes: calculateKeyframeSimilarity(uploadedAnalysis.keyframes, existingAnalysis.keyframes),
        visualEmbedding: calculateEmbeddingSimilarity(uploadedAnalysis.visualEmbedding, existingAnalysis.visualEmbedding),
        shotBoundaries: calculateBoundarySimilarity(uploadedAnalysis.shotBoundaries, existingAnalysis.shotBoundaries)
    };

    // Calculate weighted average
    let totalSimilarity = 0;
    let totalWeight = 0;

    Object.entries(weights).forEach(([key, weight]) => {
        totalSimilarity += similarities[key] * weight;
        totalWeight += weight;
    });

    return {
        overallSimilarity: Math.round(totalSimilarity / totalWeight),
        detailedScores: similarities
    };
}

// Analyze endpoint with proper error handling
// In server.js, update the analyze endpoint
// In server.js

// In server.js update the /api/analyze endpoint
videoAnalysisRouter.get('/videos', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_BACKEND_URL}/api/videos`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching videos from Python backend:', error);
        
        // Fallback to Node.js implementation if Python backend is down
        const videosDir = path.join(__dirname, '..', 'public', 'original_videos');
        const videoFiles = fs.readdirSync(videosDir)
            .filter(file => file.match(/\.(mp4|mov|avi|mkv)$/i))
            .map(filename => {
                const filePath = path.join(videosDir, filename);
                const stats = fs.statSync(filePath);
                return {
                    filename: filename,
                    url: `/original_videos/${filename}`,
                    size: stats.size,
                    uploadDate: stats.mtime
                };
            });

        res.json(videoFiles);
    }
});

// In server.js, update this part
// Updated analyze endpoint with better error handling
videoAnalysisRouter.post('/analyze', upload.single('video'), async (req, res) => {
    if (!req.file) {
        return res.status(200).json({ 
            error: false,
            filename: `${Date.now()}-unknown.mp4`,
            unique_percentage: 100,
            is_unique: true,
            similar_videos: [],
            analysis_details: { 
                error: 'No video file provided',
                note: 'Fallback analysis mode'
            }
        });
    }

    try {
        // Create form data for Python backend
        const formData = new FormData();
        const fileStream = fs.createReadStream(req.file.path);
        formData.append('video', fileStream, req.file.filename);

        console.log(`Sending video ${req.file.filename} to Python backend`);

        // Update to use Python backend's actual port
        const response = await axios.post('http://localhost:3001/api/analyze', formData, {
            headers: {
                ...formData.getHeaders(),
                'Accept': 'application/json'
            },
            timeout: 520000, // 8.7 minutes timeout
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });

        // Extensive logging
        console.log(`Python backend response for ${req.file.filename}:`, 
            typeof response.data === 'object' 
                ? JSON.stringify(response.data).substring(0, 500) 
                : 'Non-JSON response'
        );

        return res.json(response.data);
    } catch (error) {
        console.error('Error analyzing video with Python backend:', error.message);
        
        // Comprehensive error handling
        return res.status(200).json({
            temp_path: req.file.path,
            filename: req.file.filename,
            unique_percentage: 100,
            is_unique: true,
            similar_videos: [],
            fallback: true,
            error: error.message,
            analysis_details: {
                error: 'Analysis service unavailable',
                file_size: req.file.size,
                file_type: req.file.mimetype
            }
        });
    }
});

// Upload video - proxy to Python backend
videoAnalysisRouter.post('/upload', express.json(), async (req, res) => {
    const { temp_path } = req.body;
    
    if (!temp_path || !fs.existsSync(temp_path)) {
        return res.status(400).json({ error: 'Temporary file not found' });
    }

    try {
        // Create form data for Python backend
        const formData = new FormData();
        formData.append('temp_path', temp_path);

        const response = await axios.post(`${PYTHON_BACKEND_URL}/api/upload`, formData, {
            headers: formData.getHeaders()
        });

        // Return the upload result
        res.json(response.data);
    } catch (error) {
        console.error('Error uploading video with Python backend:', error);
        
        // Fallback to Node.js implementation
        try {
            // Get the filename from the temp path
            const filename = path.basename(temp_path);
            
            // Move file to permanent storage
            const finalPath = path.join(__dirname, '..', 'public', 'original_videos', filename);
            fs.renameSync(temp_path, finalPath);
            
            res.json({
                success: true,
                message: 'Video uploaded successfully (Node.js fallback)',
                filename: filename
            });
        } catch (e) {
            res.status(500).json({
                error: 'Upload failed',
                details: e.message
            });
        }
    }
});

// Test connection to Python backend
videoAnalysisRouter.get('/check-python-backend', async (req, res) => {
    try {
        await axios.get(`${PYTHON_BACKEND_URL}/api/videos`);
        res.json({ status: 'connected', backend: 'python' });
    } catch (error) {
        res.json({ status: 'disconnected', backend: 'node', error: error.message });
    }
});
// In server.js, update the /api/analyze endpoint:

// Complete implementation for the direct-analyze endpoint in server.js
// app.post('/api/direct-analyze', async (req, res) => {
//     // Store uploaded file reference
//     let uploadedFile = null;
    
//     try {
//       console.log("Direct analyze endpoint called");
      
//       // Process file upload with proper error handling
//       try {
//         await new Promise((resolve, reject) => {
//           upload.single('video')(req, res, function(err) {
//             if (err) {
//               console.error('Upload error:', err);
//               return reject(err);
//             }
//             uploadedFile = req.file;
//             resolve();
//           });
//         });
//       } catch (uploadError) {
//         console.error('File upload failed:', uploadError);
//         return res.status(200).json({ 
//           error: false,
//           fallback: true,
//           filename: `${Date.now()}-unknown.mp4`,
//           unique_percentage: 100,
//           is_unique: true,
//           similar_videos: [],
//           analysis_details: { error: 'File upload failed: ' + uploadError.message }
//         });
//       }
  
//       // Validate that we received a file
//       if (!uploadedFile) {
//         console.error('No file received in the request');
//         return res.status(200).json({ 
//           error: false,
//           fallback: true,
//           filename: `${Date.now()}-unknown.mp4`,
//           unique_percentage: 100,
//           is_unique: true,
//           similar_videos: [],
//           analysis_details: { error: 'No file received in the request' }
//         });
//       }
  
//       console.log(`File received: ${uploadedFile.filename}, size: ${uploadedFile.size} bytes`);
  
//       // Forward to Python backend for analysis
//       try {
//         // Create form data for Python backend
//         const formData = new FormData();
//         const fileStream = fs.createReadStream(uploadedFile.path);
//         formData.append('video', fileStream, uploadedFile.filename);
  
//         console.log(`Forwarding to Python backend at ${PYTHON_BACKEND_URL}/api/analyze`);
        
//         // Send to Python backend with proper timeout and error handling
//         const response = await axios.post(`${PYTHON_BACKEND_URL}/api/analyze`, formData, {
//           headers: formData.getHeaders(),
//           timeout: 120000, // 2-minute timeout
//           maxContentLength: Infinity,
//           maxBodyLength: Infinity
//         });
  
//         console.log("Received response from Python backend:", JSON.stringify(response.data));
  
//         // Return the analysis result directly
//         return res.json(response.data);
//       } catch (analysisError) {
//         // Handle analysis errors with detailed logging
//         console.error('Analysis error:', analysisError.message);
        
//         // Try to extract data from error response if available
//         let errorData = null;
//         try {
//           if (analysisError.response && analysisError.response.data) {
//             errorData = analysisError.response.data;
//             console.log("Error response data:", JSON.stringify(errorData));
//           }
//         } catch (parseError) {
//           console.error("Failed to parse error response:", parseError);
//         }
        
//         // Create a detailed fallback result
//         const fallbackResult = {
//           fallback: true,
//           temp_path: uploadedFile.path,
//           filename: uploadedFile.filename,
//           unique_percentage: 100,
//           is_unique: true,
//           similar_videos: [],
//           analysis_details: { 
//             error: 'Analysis service unavailable: ' + analysisError.message,
//             file_size: uploadedFile.size,
//             file_type: uploadedFile.mimetype
//           }
//         };
        
//         // If we extracted data from the error, include it
//         if (errorData) {
//           Object.assign(fallbackResult, errorData);
//         }
        
//         return res.status(200).json(fallbackResult);
//       }
//     } catch (outerError) {
//       // Catch-all error handler to ensure we always respond
//       console.error('Critical error in /api/direct-analyze:', outerError);
      
//       return res.status(200).json({
//         fallback: true,
//         critical_error: true,
//         filename: uploadedFile ? uploadedFile.filename : `${Date.now()}-error.mp4`,
//         temp_path: uploadedFile ? uploadedFile.path : null,
//         unique_percentage: 100,
//         is_unique: true,
//         similar_videos: [],
//         analysis_details: { 
//           error: 'Server error occurred: ' + outerError.message,
//           stack: process.env.NODE_ENV === 'development' ? outerError.stack : undefined
//         }
//       });
//     }
//   });
  
//   // Add error handlers to prevent server crashes
//   process.on('uncaughtException', (err) => {
//     console.error('Uncaught exception:', err);
//     // Keep the server running
//   });
  
//   process.on('unhandledRejection', (reason, promise) => {
//     console.error('Unhandled rejection at:', promise, 'reason:', reason);
//     // Keep the server running
//   });

  // Remove the duplicate implementation and keep only this one
// Enhanced direct-analyze endpoint with integrated analysis
app.post('/api/analyze', async (req, res) => {
    let uploadedFile = null;
    
    try {
      console.log("Analysis endpoint called");
      
      // Process file upload
      try {
        await new Promise((resolve, reject) => {
          upload.single('video')(req, res, function(err) {
            if (err) {
              console.error('Upload error:', err);
              return reject(err);
            }
            uploadedFile = req.file;
            resolve();
          });
        });
      } catch (uploadError) {
        console.error('File upload failed:', uploadError);
        return res.status(200).json({ 
          fallback: true,
          filename: `${Date.now()}-unknown.mp4`,
          unique_percentage: 100,
          is_unique: true,
          similar_videos: [],
          analysis_details: { error: 'File upload failed: ' + uploadError.message }
        });
      }
      
      if (!uploadedFile) {
        return res.status(200).json({ 
          fallback: true,
          filename: `${Date.now()}-unknown.mp4`,
          unique_percentage: 100,
          is_unique: true,
          similar_videos: [],
          analysis_details: { error: 'No file received' }
        });
      }
      
      console.log(`File received: ${uploadedFile.filename}, size: ${uploadedFile.size} bytes`);
      
      // Perform direct analysis using our own functions
      try {
        // Get list of existing videos
        const videoFiles = fs.readdirSync(directories.videos)
          .filter(file => file.match(/\.(mp4|mov|avi|mkv)$/i));
        
        // Extract basic properties of uploaded video
        const uploadedPath = uploadedFile.path;
        
        // Similar videos tracking
        const similarVideos = [];
        const similarityScores = [];
        
        // For each existing video, compare with uploaded video
        for (const existingFile of videoFiles) {
          const existingPath = path.join(directories.videos, existingFile);
          
          try {
            // Compare videos
            const comparison = await calculateVideoSimilarity(uploadedPath, existingPath);
            const similarity = comparison.overallSimilarity;
            
            // Add to similar videos if similarity is above threshold
            if (similarity > 50) {  // Increased from 40% to 50%
              // Apply a curve to increase contrast between similar and different videos
                let adjustedSimilarity = similarity;
                
                // Boost high scores (likely duplicates)
                if (similarity > 75) {
                    adjustedSimilarity = 75 + (similarity - 75) * 1.3;
                }
                // Reduce mid-range scores (likely false positives)
                else if (similarity > 55 && similarity <= 75) {
                    adjustedSimilarity = 55 + (similarity - 55) * 0.8;
                }
                
                // Cap at 100
                adjustedSimilarity = Math.min(100, adjustedSimilarity);
                
                similarityScores.push(adjustedSimilarity);
                similarVideos.push({
                    filename: existingFile,
                    overallSimilarity: Math.round(adjustedSimilarity), // Round for cleaner display
                    size: fs.statSync(existingPath).size,
                    detailed_scores: comparison.detailedScores
                });
            }
          } catch (compError) {
            console.error(`Error comparing with ${existingFile}:`, compError);
            // Continue with next video
            continue;
          }
        }
        
        // Calculate uniqueness
        let uniquePercentage = 100;
        if (similarityScores.length > 0) {
          const maxSimilarity = Math.max(...similarityScores);
          uniquePercentage = 100 - maxSimilarity;
        }
        
        // Sort similar videos by similarity
        similarVideos.sort((a, b) => b.overallSimilarity - a.overallSimilarity);
        
        // Analyze video content
        const analysis_details = {
          // Add basic video info
          resolution: "Analyzing...",
          duration: "Analyzing...",
          frames: "Analyzing..."
        };
        
        // Prepare result
        const result = {
          temp_path: uploadedFile.path,
          filename: uploadedFile.filename,
          unique_percentage: uniquePercentage,
          is_unique: uniquePercentage >= 65,
          similar_videos: similarVideos,
          analysis_details: analysis_details,
          manipulations: {
            detected: false,
            types: []
          }
        };
        
        latestAnalysisResult = {...result};
        console.log("✅ Analysis result stored for AI agent:", result.filename);

        return res.json(result);
        
      } catch (analysisError) {
        console.error('Error in analysis:', analysisError);
        return res.status(200).json({
          fallback: true,
          temp_path: uploadedFile.path,
          filename: uploadedFile.filename,
          unique_percentage: 100,
          is_unique: true,
          similar_videos: [],
          analysis_details: { 
            error: 'Analysis simplified due to error: ' + analysisError.message,
            file_size: uploadedFile.size,
            file_type: uploadedFile.mimetype
          }
        });
      }
    } catch (outerError) {
      console.error('Critical error in analysis endpoint:', outerError);
      return res.status(200).json({
        fallback: true,
        critical_error: true,
        filename: uploadedFile ? uploadedFile.filename : `${Date.now()}-error.mp4`,
        temp_path: uploadedFile ? uploadedFile.path : null,
        unique_percentage: 100,
        is_unique: true,
        similar_videos: [],
        analysis_details: { error: 'Server error occurred: ' + outerError.message }
      });
    }
  });
// Separate endpoint for final upload
app.post('/api/upload', async (req, res) => {
    try {
        console.log('1. Starting upload process');
        
        await new Promise((resolve, reject) => {
            upload.single('video')(req, res, function(err) {
                if (err) {
                    reject(err);
                    return;
                }
                resolve();
            });
        });

        if (!req.file) {
            throw new Error('No video file received');
        }

        console.log('2. Moving file to permanent storage');
        
        // Move file to permanent storage
        const finalPath = path.join(directories.videos, req.file.filename);
        fs.renameSync(req.file.path, finalPath);

        console.log('3. Upload complete:', {
            from: req.file.path,
            to: finalPath
        });

        res.json({
            success: true,
            message: 'Video uploaded successfully',
            filename: req.file.filename
        });

    } catch (error) {
        console.error('Upload error:', error);
        res.status(500).json({
            error: 'Upload failed',
            details: error.message
        });
    }
});

app.use('/api/llama', llamaRouter);

const testPythonBackendConnection = async () => {
    try {
      console.log(`Testing connection to Python backend at ${PYTHON_BACKEND_URL}...`);
      const response = await axios.get(`${PYTHON_BACKEND_URL}/api/videos`, { 
        timeout: 10000,
        headers: {
          'Accept': 'application/json'
        }
      });
      console.log('Python backend connected successfully!');
      console.log(`Found ${response.data.length} videos in database`);
      return true;
    } catch (error) {
      console.error('Failed to connect to Python backend:');
      if (error.code === 'ECONNREFUSED') {
        console.error('  - Connection refused. Is the Python server running?');
      } else if (error.code === 'ETIMEDOUT') {
        console.error('  - Connection timed out. Check network configuration.');
      } else {
        console.error(`  - ${error.message}`);
      }
      console.log('Falling back to Node.js implementation.');
      return false;
    }
  };

// Initialize server
const startServer = async () => {
    try {
        // Initialize directories
        initializeDirectories();
        const pythonBackendAvailable = await testPythonBackendConnection();
    
        // Output which implementation will be used
        if (pythonBackendAvailable) {
            console.log('Using Python backend for video analysis');
        } else {
            console.log('Using Node.js fallback for video analysis');
        }
        // Verify FFmpeg installation
        await new Promise((resolve, reject) => {
            exec('ffmpeg -version', (error, stdout) => {
                if (error) {
                    reject(new Error('FFmpeg not properly installed'));
                }
                resolve(stdout);
            });
        });

        // Start server
        app.listen(PORT, () => {
            console.log(`Server running on port ${PORT}`);
            console.log('Video analysis service ready');
        });
    } catch (error) {
        console.error('Server startup failed:', error);
        process.exit(1);
    }
};

// Start the server
startServer();



// Helper functions for similarity calculations
function calculateHashSimilarity(hash1, hash2) {
    const differences = hash1.map((h1, i) => h1 === hash2[i] ? 0 : 1)
        .reduce((sum, diff) => sum + diff, 0);
    return 100 - (differences / hash1.length * 100);
}

function calculateHistogramSimilarity(hist1, hist2) {
    const maxDiff = Math.max(hist1, hist2);
    const similarity = 100 - (Math.abs(hist1 - hist2) / maxDiff * 100);
    return Math.max(0, Math.min(100, similarity));
}

function calculateObjectSimilarity(objects1, objects2) {
    const commonObjects = objects1.filter(obj => objects2.includes(obj));
    const totalObjects = new Set([...objects1, ...objects2]).size;
    return (commonObjects.length / totalObjects) * 100;
}

function calculateKeyframeSimilarity(keyframes1, keyframes2) {
    const similar = keyframes1.filter(kf1 => 
        keyframes2.some(kf2 => path.basename(kf1) === path.basename(kf2))
    );
    return (similar.length / Math.max(keyframes1.length, keyframes2.length)) * 100;
}

function calculateEmbeddingSimilarity(embedding1, embedding2) {
    const differences = embedding1.map((e1, i) => {
        const e2 = embedding2[i];
        return Math.abs(e1.r - e2.r) + Math.abs(e1.g - e2.g) + Math.abs(e1.b - e2.b);
    });
    const avgDiff = differences.reduce((sum, diff) => sum + diff, 0) / differences.length;
    return Math.max(0, 100 - (avgDiff / 765 * 100));
}

function calculateBoundarySimilarity(boundaries1, boundaries2) {
    const countDiff = Math.abs(boundaries1.length - boundaries2.length);
    const maxCount = Math.max(boundaries1.length, boundaries2.length);
    return maxCount === 0 ? 100 : 100 - (countDiff / maxCount * 100);
}