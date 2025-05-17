import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../components/Header';
import Navigation from '../components/Navigation';


const retry = async (operation, retries = 3, delay = 1000) => {
  try {
    return await operation();
  } catch (error) {
    if (retries <= 0) {
      throw error;
    }
    
    console.log(`Operation failed, retrying in ${delay}ms... (${retries} retries left)`);
    await new Promise(resolve => setTimeout(resolve, delay));
    return retry(operation, retries - 1, delay * 1.5); // Exponential backoff
  }
};

const fetchWithTimeout = async (url, options = {}, timeout = 300000) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      credentials: 'include',
      signal: controller.signal,
      headers: {
        ...options.headers,
        'Origin': window.location.origin,
        'Referer': window.location.href
      }
    });
    
    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Detailed Fetch Error:', {
        status: response.status,
        statusText: response.statusText,
        body: errorText
      });
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
    }

    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    
    if (error.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeout/1000} seconds`);
    }
    
    throw error;
  }
};

const handleApiError = (error, fallbackMessage = 'An unexpected error occurred') => {
  console.error('API Error:', error);
  
  let userMessage = fallbackMessage;
  
  if (error.name === 'AbortError') {
    userMessage = 'The request timed out. The server might be busy.';
  } else if (error.message && !error.message.includes('TypeError')) {
    userMessage = error.message;
  } else if (error.statusCode === 413) {
    userMessage = 'The video file is too large to upload.';
  } else if (error.statusCode >= 500) {
    userMessage = 'The server encountered an error. Please try again later.';
  } else if (!navigator.onLine) {
    userMessage = 'You appear to be offline. Please check your internet connection.';
  }
  
  return userMessage;
};


function Upload() {
  const navigate = useNavigate();
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [warningMessage, setWarningMessage] = useState('');

  
  // Handle file selection
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file size
      if (file.size > 200 * 1024 * 1024) {  // 200 MB limit
        setErrorMessage('File size exceeds 200MB limit');
        return;
      }
      
      // Validate file type
      const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska'];
      if (!validTypes.includes(file.type)) {
        setWarningMessage('File type may not be supported. Try using MP4, MOV, AVI, or MKV format.');
      } else {
        setWarningMessage('');
      }
      
      // Create the URL immediately
      try {
        const url = URL.createObjectURL(file);
        console.log("Created video URL:", url);
        
        setVideoFile(file);
        setVideoUrl(url);
        setErrorMessage('');
      } catch (error) {
        console.error("Error creating object URL:", error);
        setErrorMessage('Could not preview this video file');
        setVideoFile(file);  // Still set the file for upload
      }
    }
  };

  // Send for analysis with improved error handling
  // In Upload.js, modify the handleAnalyze function:

  const handleAnalyze = async () => {
    console.group('🔍 Video Analysis Debug');
    console.time('Analysis Duration');
    
    try {
      console.log('📤 Initiating Video Upload');
      
      const formData = new FormData();
      formData.append('video', videoFile);
      
      // More detailed request logging
      console.log('Request Details:', {
        url: 'http://localhost:3001/api/analyze',
        method: 'POST',
        credentials: 'include'
      });
      
      const response = await fetchWithTimeout(
        'http://localhost:3001/api/analyze', 
        {
          method: 'POST',
          body: formData,
          headers: {
            'Accept': 'application/json'
          }
        }, 
        600000  // 10-minute timeout
      );
      
      console.log('📥 Response Status:', response.status);
      
      const result = await response.json();
      console.log('✅ Analysis Result:', result);
      
      // Existing logic for handling successful analysis
      localStorage.setItem('analysisResult', JSON.stringify(result));
      localStorage.setItem('videoPreviewUrl', videoUrl);
      
      navigate('/analysis-results');
    } catch (error) {
      console.error('❌ Analysis Request Error:', error);
      
      const userMessage = handleApiError(error, 'Video analysis failed');
      setErrorMessage(userMessage);
    } finally {
      console.timeEnd('Analysis Duration');
      console.groupEnd();
    }
  };

  // Simulate progress for better UX
  const startProgressSimulation = () => {
    const totalSteps = 20;
    const maxProgress = 95;
    const interval = 500;
    
    let currentStep = 0;
    
    const progressInterval = setInterval(() => {
      currentStep++;
      const simulatedProgress = Math.min(
        Math.round((currentStep / totalSteps) * maxProgress), 
        maxProgress
      );
      
      setAnalysisProgress(simulatedProgress);
      
      if (currentStep >= totalSteps) {
        clearInterval(progressInterval);
      }
    }, interval);
    
    return progressInterval;
  };

  // Function to skip analysis and go directly to upload
  const handleSkipAnalysis = () => {
    if (!videoFile) {
      setErrorMessage('Please select a video first');
      return;
    }
    
    // Create a basic result object
    const basicResult = {
      temp_path: null, // We won't have a temp path
      filename: `${Date.now()}-${videoFile.name}`,
      unique_percentage: 100,
      is_unique: true,
      similar_videos: [],
      analysis_details: {
        note: "Analysis was skipped"
      },
      skipped: true
    };
    
    // Store in localStorage
    localStorage.setItem('analysisResult', JSON.stringify(basicResult));
    if (videoUrl) {
      localStorage.setItem('videoPreviewUrl', videoUrl);
    }
    
    // Navigate to results with a flag indicating direct upload
    navigate('/analysis-results', { state: { directUpload: true } });
  };

  return (
    <div style={{ backgroundColor: '#000', color: 'white', minHeight: '100vh' }}>
      <Header />
      
      <div style={{ maxWidth: '800px', margin: '0 auto', padding: '20px' }}>
        <h2>Upload Video</h2>
        
        {/* File input - always visible */}
        <div style={{ marginBottom: '20px' }}>
          <input 
            type="file" 
            accept="video/*" 
            onChange={handleFileChange}
            style={{ color: 'white', marginBottom: '10px' }} 
          />
        </div>
        
        {/* Warning message */}
        {warningMessage && (
          <div style={{ 
            padding: '10px', 
            backgroundColor: '#856404', 
            color: '#fff3cd',
            borderRadius: '4px', 
            marginBottom: '15px' 
          }}>
            ⚠️ {warningMessage}
          </div>
        )}
        
        {/* Video preview - appears when videoUrl exists */}
        {videoFile && (
          <div style={{ marginBottom: '20px', border: '1px solid #333', padding: '10px', borderRadius: '8px' }}>
            {videoUrl ? (
              <video 
                src={videoUrl} 
                controls 
                style={{ width: '100%', borderRadius: '8px', marginBottom: '15px' }} 
              />
            ) : (
              <div style={{
                width: '100%',
                height: '240px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: '#333',
                borderRadius: '8px',
                marginBottom: '15px'
              }}>
                <p>Preview not available for this file format</p>
              </div>
            )}
            
            {/* Analysis progress bar */}
            {isLoading && (
              <div style={{ marginBottom: '15px' }}>
                <div style={{ 
                  width: '100%', 
                  height: '6px', 
                  backgroundColor: '#333', 
                  borderRadius: '3px',
                  overflow: 'hidden',
                  marginBottom: '8px'
                }}>
                  <div 
                    style={{
                      height: '100%',
                      width: `${analysisProgress}%`,
                      backgroundColor: '#4CAF50',
                      transition: 'width 0.5s ease-in-out'
                    }}
                  />
                </div>
                <div style={{ fontSize: '14px', color: '#aaa' }}>
                  Analyzing video: {analysisProgress}% complete
                </div>
              </div>
            )}
            
            {/* Action buttons */}
            <div style={{ display: 'flex', gap: '10px' }}>
              {/* Analyze button */}
              <button 
                onClick={handleAnalyze}
                disabled={isLoading}
                style={{
                  padding: '12px 24px',
                  backgroundColor: isLoading ? '#666' : '#6366f1',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: isLoading ? 'not-allowed' : 'pointer',
                  flex: '1',
                  fontSize: '16px'
                }}
              >
                {isLoading ? 'Analyzing...' : 'Analyze Video'}
              </button>
              
              {/* Skip Analysis button */}
              {!isLoading && (
                <button 
                  onClick={handleSkipAnalysis}
                  style={{
                    padding: '12px 24px',
                    backgroundColor: '#374151',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '16px'
                  }}
                >
                  Skip Analysis
                </button>
              )}
            </div>
          </div>
        )}
        
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
      </div>
      
      <Navigation />
    </div>
  );
}

// Add the default export statement
export default Upload;