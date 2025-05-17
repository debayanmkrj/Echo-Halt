import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navigation from './components/Navigation';
import Home from './pages/Home';
import Upload from './pages/Upload';
import { AnalysisResults } from './pages/AnalysisResults';
import VideoView from './pages/VideoView';
import ChatAgent from './pages/ChatAgent';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app-container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/analysis-results" element={<AnalysisResults />} />
          <Route path="/video/:id" element={<VideoView />} />
          <Route path="/chat" element={<ChatAgent />} />
        </Routes>
        <Navigation />
      </div>
    </Router>
  );
}

export default App;