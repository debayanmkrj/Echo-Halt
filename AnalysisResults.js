import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import Header from '../components/Header';
import Navigation from '../components/Navigation';
// Add these imports at the top of your AnalysisResults.js file
//import { fetchWithTimeout, fetchJsonWithTimeout, retry, handleApiError } from 'C:/Users/dbmkr/Documents/AME 598- AI Social Good/Assignment - Presentation Proposal/echo-hall/src/utilities/fetchWithTimeout.js'; // Adjust the path as needed
import { fetchWithTimeout, fetchJsonWithTimeout, retry, handleApiError } from '../utilities/fetchWithTimeout';

export function AnalysisResults() {
  const navigate = useNavigate();
  const location = useLocation();
  const [analysisResult, setAnalysisResult] = useState(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [caption, setCaption] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [loadingData, setLoadingData] = useState(true);
  
  // Determine if this is a direct upload (skipped analysis)
  const isDirectUpload = location.state && location.state.directUpload;
// Add this with your other state variables
  const [progressMessage, setProgressMessage] = useState("Preparing upload...");
  // Load analysis data from localStorage when component mounts
  useEffect(() => {
    console.log("AnalysisResults component mounted");
    setLoadingData(true);
    
    try {
      // Get analysis result from localStorage
      const storedResult = localStorage.getItem('analysisResult');
      console.log("Stored result from localStorage:", storedResult ? storedResult.substring(0, 100) + "..." : "none");
      
      if (storedResult) {
        try {
          const parsedResult = JSON.parse(storedResult);
          console.log("Successfully parsed analysis result:", parsedResult);
          setAnalysisResult(parsedResult);
        } catch (parseError) {
          console.error("Error parsing stored analysis result:", parseError);
          setErrorMessage("Failed to load analysis data. Please try again.");
        }
      } else if (location.state && location.state.analysisResult) {
        // Alternative: get from location state
        console.log("Using analysis result from location state");
        setAnalysisResult(location.state.analysisResult);
      } else {
        console.warn("No analysis result found in localStorage or location state");
      }
      
      // Get video URL from localStorage
      const storedVideoUrl = localStorage.getItem('videoPreviewUrl');
      if (storedVideoUrl) {
        console.log("Retrieved video URL from localStorage");
        setVideoUrl(storedVideoUrl);
      }
    } catch (error) {
      console.error("Error initializing analysis result:", error);
      setErrorMessage(`Error loading analysis data: ${error.message}`);
    } finally {
      setLoadingData(false);
    }
  }, [location.state]);

  // Debug log whenever analysisResult changes
  // Add this after your existing state variables declarations
useEffect(() => {
  // Enhanced data inspection for debugging
  if (analysisResult) {
    console.group('Analysis Result Inspection');
    console.log('Raw result:', analysisResult);
    console.log('Uniqueness percentage:', analysisResult.unique_percentage);
    console.log('Is unique:', analysisResult.is_unique);
    console.log('Fallback used:', analysisResult.fallback);
    console.log('Similar videos count:', analysisResult.similar_videos?.length || 0);
    
    // Log manipulation details
    if (analysisResult.manipulations) {
      console.log('Manipulations detected:', analysisResult.manipulations.detected);
      console.log('Manipulation types:', analysisResult.manipulations.types);
    }
    
    if (analysisResult.analysis_details) {
      console.log('Available algorithm scores:', Object.entries(analysisResult.analysis_details)
        .filter(([key, value]) => 
          ["perceptual_hash", "histogram", "deep_features", 
           "scene_structure", "motion", "manipulation", "keyframes"].includes(key))
        .map(([key, value]) => `${key}: ${value}%`)
      );
    }
    
    console.groupEnd();
  }
}, [analysisResult]);

// Replace your current handleUpload function with this one
const handleUpload = async () => {
  if (!analysisResult) {
    setErrorMessage('No analysis data available');
    return;
  }
  
  setIsUploading(true);
  setUploadProgress(0);
  setErrorMessage('');
  
  try {
    // Start progress simulation
    const progressInterval = simulateProgress();
    
    let formData = new FormData();
    
    // Ensure we're using the correct temp_path
    if (analysisResult.temp_path) {
      formData.append('temp_path', analysisResult.temp_path);
      formData.append('caption', caption);
      console.log("Uploading with temp_path:", analysisResult.temp_path);
      
      // Specify target directory explicitly
      formData.append('target_dir', 'C:\\Users\\dbmkr\\Documents\\AME 598- AI Social Good\\Assignment - Presentation Proposal\\echo-hall\\echo-hall\\server\\public\\original_videos');
    } else {
      clearInterval(progressInterval);
      setIsUploading(false);
      setErrorMessage('No file path available for upload');
      return;
    }
    
    // Send the upload request
    console.log("Sending upload request to server");
    const response = await fetchWithTimeout('http://localhost:3001/api/upload', {
      method: 'POST',
      body: formData
    }, 300000); // 5-minute timeout
    
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }
    
    const uploadResult = await response.json();
    
    // Clear progress interval
    clearInterval(progressInterval);
    setUploadProgress(100);
    
    console.log("Upload result:", uploadResult);
    
    setUploadSuccess(true);
    
    // Navigate to video view page after short delay
    setTimeout(() => {
      navigate(`/video/${encodeURIComponent(uploadResult.filename)}`);
    }, 2000);
    
  } catch (error) {
    console.error('Upload error:', error);
    
    const userFriendlyError = handleApiError(error, 'Failed to upload video. Please try again.');
    setErrorMessage(userFriendlyError);
    setIsUploading(false);
  }
};

  // Replace your current simulateProgress function with this one
  const simulateProgress = () => {
    const totalSteps = 20;
    const maxProgress = 95;
    const interval = 400;
    
    // Create an array of messages for different progress stages
    const progressMessages = [
      { threshold: 10, message: "Initializing upload..." },
      { threshold: 30, message: "Transferring video to server..." },
      { threshold: 50, message: "Processing video..." },
      { threshold: 70, message: "Finalizing upload..." },
      { threshold: 90, message: "Almost done..." }
    ];
    
    let currentStageIndex = 0;
    let step = 0;
    
    const progressInterval = setInterval(() => {
      step++;
      const simulatedProgress = Math.min(
        Math.round((step / totalSteps) * maxProgress), 
        maxProgress
      );
      
      setUploadProgress(simulatedProgress);
      
      // Update message based on progress
      if (currentStageIndex < progressMessages.length && 
          simulatedProgress >= progressMessages[currentStageIndex].threshold) {
        setProgressMessage(progressMessages[currentStageIndex].message);
        currentStageIndex++;
      }
      
      if (step >= totalSteps) {
        clearInterval(progressInterval);
      }
    }, interval);
    
    return progressInterval;
  };

  const handleCancel = () => {
    navigate('/upload');
  };

  const calculateAverageAlgorithmScore = (analysisDetails) => {
    if (!analysisDetails) return 0;
    
    const algorithmKeys = [
      "perceptual_hash", "histogram", "deep_features", 
      "scene_structure", "motion", "keyframes"
    ];
    
    let totalScore = 0;
    let count = 0;
    
    // Sum up all available algorithm scores
    algorithmKeys.forEach(key => {
      const score = analysisDetails[key];
      if (typeof score === 'number' && !isNaN(score)) {
        totalScore += score;
        count++;
      }
    });
    
    // Return average, or 0 if no scores found
    return count > 0 ? totalScore / count : 0;
  };
  // Helper function to render uniqueness meter
  const renderUniquenessMeter = () => {
    if (!analysisResult) return null;
    
    // Get the uniqueness percentage from the analysis result
    const percentage = typeof analysisResult.unique_percentage === 'number' 
      ? analysisResult.unique_percentage 
      : 100;
      
    const isUnique = analysisResult.is_unique !== false; // Default to true if not explicitly false
    const isFirstVideo = analysisResult.first_video;
    const isFallback = analysisResult.fallback;
    const hasManipulations = analysisResult.manipulations?.detected;
    const avgAlgorithmScore = calculateAverageAlgorithmScore(analysisResult?.analysis_details);
    const confidenceConflict = avgAlgorithmScore > 80 && percentage < 60;


    let message;
    if (isFirstVideo) {
      message = "This is the first video in the database, so it's automatically considered unique.";
    } else if (isFallback) {
      message = "Analysis was limited. Basic uniqueness estimation only.";
      // Add error details if available
      if (analysisResult.analysis_details?.error) {
        message += ` (${analysisResult.analysis_details.error})`;
      }
    } else if (hasManipulations) {
      message = isUnique 
        ? "This video appears unique, but has been modified from its original form."
        : "This video has similarities to existing content and shows signs of manipulation.";
    } else if (isUnique) {
      message = "This video appears to be unique compared to existing content.";
    } else {
      message = "This video has similarities to existing content in our database.";

      if (confidenceConflict) {
        message += " Note: Individual algorithm scores suggest this may actually be unique content.";
      }
    }
    
    return (
      <div style={{ marginBottom: '30px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
          <span>Uniqueness Score</span>
          <span>{percentage}%</span>
        </div>
        
        <div style={{ 
          height: '8px', 
          backgroundColor: '#333',
          borderRadius: '4px',
          overflow: 'hidden',
          marginBottom: '10px'
        }}>
          <div style={{ 
            height: '100%',
            width: `${percentage}%`,
            backgroundColor: confidenceConflict 
              ? '#FFA500' // Orange for conflicting scores
              : (hasManipulations 
                  ? '#FF9800' // Orange for manipulated videos
                  : (isUnique ? '#4CAF50' : '#f44336')),
            borderRadius: '4px',
            transition: 'width 0.5s ease-out'
          }}></div>
        </div>
        
        <p style={{ 
          color: confidenceConflict ? '#FFA500' : '#aaa', 
          fontStyle: 'italic',
          fontWeight: confidenceConflict ? 'bold' : 'normal'
        }}>
          {message}
        </p>
      </div>
    );
  };

  // Add this function inside your AnalysisResults component
  const renderManipulationInfo = () => {
    if (!analysisResult || !analysisResult.manipulations) return null;
    
    const { detected, types = [] } = analysisResult.manipulations;
    
    if (!detected || types.length === 0) return null;
    
    // Map technical terms to user-friendly descriptions with clearer explanations
    const manipulationDescriptions = {
      'frame_insertion': 'Content has been added at the beginning of the video',
      'temporal_truncation': 'The original video has been shortened or cut',
      'overlay_modification': 'Text or graphics have been added on top of the original video',
      'split_screen_composition': 'The video uses a split-screen effect with multiple video sources',
      'reflection_transformation': 'The video has been flipped horizontally (mirrored)'
    };
    
    return (
      <div style={{ 
        backgroundColor: '#FF6B6B', 
        color: 'white', 
        padding: '15px', 
        borderRadius: '6px',
        marginBottom: '20px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
      }}>
        <h3 style={{ margin: '0 0 10px 0' }}>⚠️ Manipulations Detected</h3>
        <ul style={{ 
          paddingLeft: '20px', 
          marginBottom: '10px' 
        }}>
          {types.map((type, index) => (
            <li key={index} style={{ marginBottom: '8px' }}>
              <strong>{manipulationDescriptions[type] || type.replace(/_/g, ' ')}</strong>
            </li>
          ))}
        </ul>
        <p style={{ 
          fontSize: '0.9rem', 
          margin: '0', 
          fontStyle: 'italic' 
        }}>
          This video appears to be a modified version of existing content.
        </p>
      </div>
    );
  };
  // Helper function to render similar videos section
  const renderSimilarVideos = () => {
    if (!analysisResult || !analysisResult.similar_videos || analysisResult.similar_videos.length === 0) {
      return null;
    }
    const similarVideo = analysisResult.similar_videos[0];
    return (
      <div style={{ marginBottom: '30px' }}>
        <h3>Most Similar Video</h3>
        <div style={{ 
          backgroundColor: '#222', 
          borderRadius: '6px', 
          padding: '15px',
          border: '1px solid #333'
        }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '10px'
          }}>
            <span style={{ 
              maxWidth: '70%', 
              overflow: 'hidden', 
              textOverflow: 'ellipsis', 
              whiteSpace: 'nowrap',
              fontSize: '1.1rem',
              fontWeight: 'bold'
            }}>
              {similarVideo.filename}
            </span>
            <span style={{ 
              color: similarVideo.overallSimilarity > 70 ? '#f44336' : '#aaa',
              fontWeight: 'bold',
              fontSize: '1.2rem',
              backgroundColor: '#333',
              padding: '4px 12px',
              borderRadius: '20px'
            }}>
              {similarVideo.overallSimilarity}% similar
            </span>
          </div>
          
          {/* Add manipulation indicator if detected */}
          {similarVideo.manipulations && similarVideo.manipulations.detected && (
            <div style={{ 
              backgroundColor: 'rgba(255, 107, 107, 0.3)', 
              padding: '10px 15px',
              borderRadius: '4px',
              marginBottom: '10px'
            }}>
              <span style={{ color: '#FF6B6B', fontWeight: 'bold' }}>
                ⚠️ Manipulated Video Detected
              </span>
              <div style={{ marginTop: '5px', fontSize: '0.9rem' }}>
                {similarVideo.manipulations.types.map(type => 
                  <div key={type}>
                    - {type.replace(/_/g, ' ')}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderAlgorithmScores = () => {
    if (!analysisResult || !analysisResult.analysis_details) return null;
    
    // List of algorithm keys to check for, now including manipulation
    const algorithms = [
      "perceptual_hash", "histogram", "deep_features", 
      "scene_structure", "motion", "manipulation", "keyframes"
    ];
    
    // Filter out algorithms that don't have a score
    const availableAlgorithms = algorithms.filter(
      algo => {
        const value = analysisResult.analysis_details[algo];
        // Check if the value is a number or can be converted to one
        return value !== undefined && !isNaN(Number(value));
      }
    );
    
    if (availableAlgorithms.length === 0) return null;
    
    // Map technical terms to user-friendly names
    const algoNames = {
      "perceptual_hash": "Visual Hash",
      "histogram": "Color Profile",
      "deep_features": "Visual Features",
      "scene_structure": "Scene Structure",
      "motion": "Motion Patterns",
      "manipulation": "Manipulation Detection",
      "keyframes": "Key Frames"
    };
    
    return (
      <div style={{ marginTop: '15px' }}>
        <h4>Algorithm Scores</h4>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', 
          gap: '10px',
          backgroundColor: '#1a1a1a',
          padding: '15px',
          borderRadius: '4px'
        }}>
          {availableAlgorithms.map(algo => {
            // Get the score, ensuring it's a number
            let score = analysisResult.analysis_details[algo];
            
            // Handle array values by taking the first number or a default
            if (Array.isArray(score)) {
              // If it's an array of values, take the first number or a default
              const numericValues = score.filter(val => !isNaN(Number(val)));
              score = numericValues.length > 0 ? Number(numericValues[0]) : 50;
            } else if (typeof score === 'string') {
              // Try to parse string values
              score = !isNaN(Number(score)) ? Number(score) : 50;
            } else if (typeof score !== 'number') {
              // Default for non-numeric values
              score = 50;
            }
            
            // Ensure the score is between 0 and 100
            score = Math.max(0, Math.min(100, score));
            
            return (
              <div key={algo}>
                <div style={{ color: '#aaa', fontSize: '0.9rem' }}>
                  {algoNames[algo] || algo.replace(/_/g, ' ')}
                </div>
                <div style={{
                  color: algo === 'manipulation' && score < 70 ? '#FF6B6B' : 'white'
                }}>
                  {Math.round(score)}%
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div style={{ backgroundColor: '#000', color: 'white', minHeight: '100vh' }}>
      <Header />
      
      <div style={{ maxWidth: '800px', margin: '0 auto', padding: '20px' }}>
        <h2>Video Analysis Results</h2>
        
        {loadingData ? (
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <p>Loading analysis data...</p>
          </div>
        ) : !analysisResult ? (
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <p>No analysis data available. Please upload a video first.</p>
            <button 
              onClick={() => navigate('/upload')}
              style={{
                padding: '12px 24px',
                backgroundColor: '#6366f1',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '16px',
                marginTop: '20px'
              }}
            >
              Go to Upload
            </button>
          </div>
        ) : (
          <>
            {/* Video preview */}
            {videoUrl && (
              <div style={{ marginBottom: '20px' }}>
                <video 
                  src={videoUrl} 
                  controls={false}  // Remove controls
                  onClick={(e) => {
                    // Toggle play/pause on click
                    const video = e.currentTarget;
                    if (video.paused) {
                      video.play();
                    } else {
                      video.pause();
                    }
                  }}
                  style={{ 
                    width: '100%', 
                    borderRadius: '8px',
                    cursor: 'pointer',  // Show pointer cursor on hover
                  }} 
                />
              </div>
            )}  
            
            {/* Analysis results box */}
            <div style={{ 
              backgroundColor: '#111', 
              borderRadius: '8px', 
              padding: '20px',
              marginBottom: '20px'
            }}>
              <h3>Analysis Results</h3>
              
              {/* Uniqueness meter */}
              {renderUniquenessMeter()}
              {renderManipulationInfo()}
              
              {/* Video details */}
              {analysisResult.analysis_details && (
                <div style={{ marginBottom: '20px' }}>
                  <h4>Video Details</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '15px' }}>
                    {analysisResult.analysis_details.resolution && (
                      <div>
                        <div style={{ color: '#aaa', fontSize: '0.9rem' }}>Resolution</div>
                        <div>{analysisResult.analysis_details.resolution}</div>
                      </div>
                    )}
                    
                    {analysisResult.analysis_details.duration !== undefined && (
                      <div>
                        <div style={{ color: '#aaa', fontSize: '0.9rem' }}>Duration</div>
                        <div>{analysisResult.analysis_details.duration} seconds</div>
                      </div>
                    )}
                    
                    {analysisResult.analysis_details.fps && (
                      <div>
                        <div style={{ color: '#aaa', fontSize: '0.9rem' }}>Frame Rate</div>
                        <div>{analysisResult.analysis_details.fps} FPS</div>
                      </div>
                    )}
                  </div>

                   {/* Add the algorithm scores section here */}
                    {renderAlgorithmScores()}
                  
                  {analysisResult.analysis_details.note && (
                    <div style={{ 
                      backgroundColor: '#292929', 
                      padding: '10px', 
                      borderRadius: '4px',
                      marginTop: '15px',
                      fontSize: '0.9rem',
                      color: '#aaa' 
                    }}>
                      Note: {analysisResult.analysis_details.note}
                    </div>
                  )}
                </div>
              )}
              
              {/* Similar videos */}
              {renderSimilarVideos()}
              
              {/* Caption input */}
              {!uploadSuccess && (
                <div style={{ marginTop: '20px', marginBottom: '20px' }}>
                  <label htmlFor="caption" style={{ display: 'block', marginBottom: '10px' }}>
                    Add Caption (Optional)
                  </label>
                  <textarea
                    id="caption"
                    value={caption}
                    onChange={(e) => setCaption(e.target.value)}
                    placeholder="Add a description for this video"
                    style={{
                      width: '100%',
                      padding: '12px',
                      borderRadius: '4px',
                      backgroundColor: '#222',
                      border: '1px solid #333',
                      color: 'white',
                      minHeight: '100px'
                    }}
                  />
                </div>
              )}
              
              {/* Action buttons or success message */}
              {uploadSuccess ? (
                <div style={{ 
                  backgroundColor: '#1b4332', 
                  color: '#d8f3dc', 
                  padding: '15px', 
                  borderRadius: '4px',
                  textAlign: 'center'
                }}>
                  <p>Video uploaded successfully!</p>
                  <p>Redirecting to video page...</p>
                </div>
              ) : (
                <div style={{ display: 'flex', gap: '15px', marginTop: '20px' }}>
                  <button
                    onClick={handleCancel}
                    disabled={isUploading}
                    style={{
                      padding: '12px',
                      backgroundColor: '#374151',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: isUploading ? 'not-allowed' : 'pointer',
                      opacity: isUploading ? 0.7 : 1,
                      width: analysisResult?.unique_percentage >= 40 ? '50%' : '100%'
                    }}
                  >
                    Cancel
                  </button>
                  
                  {/* Only show Confirm Upload button if uniqueness is at least 70% */}
                  {analysisResult && analysisResult.unique_percentage >= 40 && (
                    <button
                      onClick={handleUpload}
                      disabled={isUploading}
                      style={{
                        padding: '12px',
                        backgroundColor: '#4CAF50',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: isUploading ? 'not-allowed' : 'pointer',
                        width: '50%',
                        position: 'relative',
                        overflow: 'hidden'
                      }}
                    >
                      {isUploading ? 'Uploading...' : 'Confirm Upload'}
                      
                      {isUploading && (
                        <div style={{
                          position: 'absolute',
                          bottom: 0,
                          left: 0,
                          height: '4px',
                          width: `${uploadProgress}%`,
                          backgroundColor: 'rgba(255, 255, 255, 0.5)',
                          transition: 'width 0.3s ease-out'
                        }}></div>
                      )}
                      {isUploading && (
                        <div style={{
                          marginTop: '10px',
                          position: 'absolute',
                          bottom: '-25px',
                          left: 0,
                          right: 0,
                          fontSize: '12px',
                          color: '#aaa',
                          textAlign: 'center'
                        }}>
                          {progressMessage}
                        </div>
                      )}
                    </button>
                  )}
                </div>
              )}
            </div>
            
            {/* Error message */}
            {errorMessage && (
              <div style={{ 
                padding: '15px', 
                backgroundColor: '#f44336', 
                borderRadius: '4px', 
                marginBottom: '20px' 
              }}>
                {errorMessage}
              </div>
            )}
          </>
        )}
      </div>
      
      <Navigation />
    </div>
  );
}

export default AnalysisResults;