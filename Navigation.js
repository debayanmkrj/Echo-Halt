import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Home, PlusSquare, MessageCircle, Play } from 'lucide-react';

function Navigation() {
  const location = useLocation();
  const navigate = useNavigate();

  const handlePlayClick = async () => {
    try {
      const response = await fetch('http://localhost:3001/api/videos');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const videos = await response.json();
      console.log('Fetched videos:', videos);
      
      if (videos && videos.length > 0) {
        const randomIndex = Math.floor(Math.random() * videos.length);
        navigate(`/video/${randomIndex}`, { 
          state: { 
            videos,
            currentIndex: randomIndex,
            fromNav: true 
          }
        });
      }
    } catch (error) {
      console.error('Error fetching videos:', error);
    }
  };

  return (
    <nav className="nav-bar">
      <Link to="/" className={location.pathname === '/' ? 'active' : ''}>
        <Home className="nav-icon" />
      </Link>
      <Link to="/upload" className={location.pathname === '/upload' ? 'active' : ''}>
        <PlusSquare className="nav-icon" />
      </Link>
      <Link to="/chat" className={location.pathname === '/chat' ? 'active' : ''}>
        <MessageCircle className="nav-icon" />
      </Link>
      <div 
        onClick={handlePlayClick} 
        className={`nav-link ${location.pathname.startsWith('/video') ? 'active' : ''}`}
        style={{ cursor: 'pointer' }}
      >
        <Play className="nav-icon" />
      </div>
    </nav>
  );
}

export default Navigation;