import React, { useRef, useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { X } from 'lucide-react';
import Navigation from '../components/Navigation';

function VideoView() {
  const navigate = useNavigate();
  const location = useLocation();
  const { id } = useParams();
  const [videos, setVideos] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const videoRef = useRef(null);
  const [touchStart, setTouchStart] = useState(null);
  const [touchEnd, setTouchEnd] = useState(null);
  const [isPlaying, setIsPlaying] = useState(true);

  // Initialize videos and current index
  useEffect(() => {
    const initializeVideos = async () => {
      try {
        if (location.state?.videos) {
          setVideos(location.state.videos);
          setCurrentIndex(parseInt(id) || 0);
        } else {
          const response = await fetch('http://localhost:3001/api/videos');
          const videoList = await response.json();
          setVideos(videoList);
          setCurrentIndex(parseInt(id) || Math.floor(Math.random() * videoList.length));
        }
      } catch (error) {
        console.error('Error initializing videos:', error);
      }
    };
    initializeVideos();
  }, [id, location.state]);

  // Auto-play handling
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.play().catch(error => {
        console.warn('Autoplay failed:', error);
      });
    }
  }, [currentIndex]);

  // Handle scroll/swipe navigation
  const minSwipeDistance = 50;

  const onTouchStart = (e) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0].clientY);
  };

  const onTouchMove = (e) => {
    setTouchEnd(e.targetTouches[0].clientY);
  };
  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };
  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return;
    const distance = touchStart - touchEnd;
    const isUpSwipe = distance > minSwipeDistance;
    const isDownSwipe = distance < -minSwipeDistance;

    if (isUpSwipe && currentIndex < videos.length - 1) {
      setCurrentIndex(prev => prev + 1);
    }
    if (isDownSwipe && currentIndex > 0) {
      setCurrentIndex(prev => prev - 1);
    }
  };

  // Handle mouse wheel scrolling
  const onWheel = useCallback((e) => {
    if (e.deltaY > 0 && currentIndex < videos.length - 1) {
      setCurrentIndex(prev => prev + 1);
    } else if (e.deltaY < 0 && currentIndex > 0) {
      setCurrentIndex(prev => prev - 1);
    }
  }, [currentIndex, videos.length]);

  const handleClose = () => navigate('/');

  const getCurrentVideoUrl = () => {
    if (!videos[currentIndex]) return '';
    if (typeof videos[currentIndex] === 'string') {
      return `http://localhost:3001/original_videos/${videos[currentIndex]}`;
    }
    return `http://localhost:3001${videos[currentIndex].url}`;
  };

  return (
    <div className="video-view-container" style={{ 
      position: 'fixed', 
      top: 0, 
      left: 0, 
      width: '100vw', 
      height: '100vh', 
      backgroundColor: '#000',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <div 
        className="reels-container"
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
        onWheel={onWheel}
        onClick={togglePlayPause}
        style={{ 
          flex: 1,
          position: 'relative',
          overflow: 'hidden',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center'
        }}
      >
        {videos.length > 0 && (
          <video
            ref={videoRef}
            className="reels-video"
            autoPlay={false}
            controls={false}
            loop
            playsInline
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain'
            }}
            src={getCurrentVideoUrl()}
          />
        )}
        
        <div className="video-controls" style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          padding: '1rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'linear-gradient(to bottom, rgba(0,0,0,0.5) 0%, transparent 100%)'
        }}>
          <button 
            className="close-button"
            onClick={handleClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'white',
              cursor: 'pointer',
              padding: '0.5rem'
            }}
          >
            <X size={24} />
          </button>
          <div className="scroll-indicator" style={{
            color: 'white',
            fontSize: '1rem'
          }}>
            {currentIndex + 1} / {videos.length}
          </div>
        </div>
      </div>
      <Navigation />
    </div>
  );
}

export default VideoView;