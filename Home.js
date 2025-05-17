import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../components/Header';
import Navigation from '../components/Navigation';
// Add this import at the top of your Home.js file
//import { fetchJsonWithTimeout, handleApiError, retry } from 'C:/Users/dbmkr/Documents/AME 598- AI Social Good/Assignment - Presentation Proposal/echo-hall/src/utilities/fetchWithTimeout.js';
// Add this import at the top of your Home.js file
import { fetchJsonWithTimeout, handleApiError, retry } from '../utilities/fetchWithTimeout';

function Home() {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    // Replace your current loadVideos function with this one
    const loadVideos = async () => {
      try {
        setLoading(true);
        
        const fetchVideos = async () => {
          return await fetchJsonWithTimeout('http://localhost:3001/api/videos', {
            method: 'GET',
            headers: {
              'Accept': 'application/json'
            }
          }, 30000); // 30-second timeout
        };
        
        // Use the retry function with the fetchVideos operation
        const data = await retry(fetchVideos, 3);
        
        console.log('Loaded videos:', data);
        setVideos(data);
        setError(null);
      } catch (error) {
        console.error('Error loading videos:', error);
        setError(handleApiError(error, 'Failed to load videos. Please refresh the page.'));
      } finally {
        setLoading(false);
      }
    };

    loadVideos();
  }, []);

  if (loading) {
    return (
      <>
        <Header />
        <div style={{ color: 'white', textAlign: 'center', marginTop: '20px' }}>
          Loading videos...
        </div>
        <Navigation />
      </>
    );
  }

  if (error) {
    return (
      <>
        <Header />
        <div style={{ color: 'red', textAlign: 'center', marginTop: '20px' }}>
          Error: {error}
        </div>
        <Navigation />
      </>
    );
  }

  return (
    <>
      <Header />
      <div className="content-grid">
        {videos.length === 0 ? (
          <div style={{ color: 'white', textAlign: 'center', gridColumn: '1/-1' }}>
            No videos found
          </div>
        ) : (
          videos.map((video, index) => (
            <div 
              key={video.filename} 
              className="grid-item"
              onClick={() => navigate(`/video/${index}`, { state: { videos, currentIndex: index } })}
            >
              <video 
                className="grid-video"
                muted
                controls={false}
                preload="metadata"
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain'
                }}
              >
                <source 
                  src={`http://localhost:3001${video.url}`} 
                  type="video/mp4" 
                />
              </video>
              <div style={{ 
                color: 'white', 
                padding: '8px',
                fontSize: '14px'
              }}>
                {video.filename}
              </div>
            </div>
          ))
        )}
      </div>
      <Navigation />
    </>
  );
}

export default Home;