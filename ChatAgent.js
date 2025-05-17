import React, { useState, useEffect, useRef } from 'react';
import Header from '../components/Header';
import axios from 'axios';
import './ChatAgent.css'; // Make sure to create this CSS file

function ChatAgent() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [llamaStatus, setLlamaStatus] = useState({ available: false, models: [] });
  const messagesEndRef = useRef(null);

  // Check Llama status on component mount
  useEffect(() => {
    checkLlamaStatus();
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Check if Llama 2 is available
  const checkLlamaStatus = async () => {
    try {
      const response = await axios.get('http://localhost:3004/api/status');
      setLlamaStatus({
        available: response.data.status === 'connected',
        models: response.data.models || []
      });
      
      // Add system message about Llama availability
      if (response.data.status === 'connected') {
        setMessages([{
          text: "Welcome! I'm your Echo Hall video analysis assistant powered by Mistral. I can answer questions about the project and video analysis capabilities.",
          type: 'assistant'
        }]);
      } else {
        setMessages([{
          text: "Welcome! I'm your Echo Hall assistant, but I couldn't connect to the Llama 2 model. Please check that Ollama is running.",
          type: 'assistant'
        }]);
      }
    } catch (error) {
      console.error('Error checking Llama status:', error);
      setLlamaStatus({ available: false, models: [] });
      setMessages([{
        text: "Welcome! I'm your Echo Hall assistant, but I couldn't connect to the Llama 2 model. Please check that Ollama is running.",
        type: 'assistant'
      }]);
    }
  };

  // Function to fetch latest analysis data
// In ChatAgent.js, replace the fetchLatestAnalysis function
const fetchLatestAnalysis = async () => {
  try {
    console.log("Attempting to fetch latest analysis data...");
    // Use the direct endpoint
    const response = await axios.get('http://localhost:3002/api/latest-analysis', {
      timeout: 5000
    });
    
    console.log("Response status:", response.status);
    
    if (response.data && !response.data.error) {
      console.log("Analysis data retrieved successfully");
      return response.data;
    } else {
      console.warn("Analysis API returned an error:", response.data?.error || "Unknown error");
      return null;
    }
  } catch (error) {
    console.error('Error fetching analysis data:', error.message);
    return null;
  }
};
  // Modify the sendMessage function in ChatAgent.js
  // In ChatAgent.js, modify the sendMessage function
// In sendMessage function in ChatAgent.js, simplify to:
const sendMessage = async () => {
  if (input.trim()) {
    // Add user message to chat
    const userMessage = { text: input, type: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    
    // Show typing indicator
    setIsTyping(true);
    
    try {
      if (llamaStatus.available) {
        // Send message to Llama 2 with a longer timeout
        const response = await axios.post('http://localhost:3004/api/chat', 
          { message: input },
          { timeout: 500000 } // second timeout
        );
        
        // Add Llama response to chat
        setMessages(prev => [...prev, {
          text: response.data.response,
          type: 'assistant'
        }]);
      } else {
        // Fallback message if Llama 2 is not available
        setMessages(prev => [...prev, {
          text: "I'm sorry, but I couldn't connect to the Mistral model. Please make sure Mistral is running with the llama2 model loaded.",
          type: 'assistant'
        }]);
      }
    } catch (error) {
      console.error('Error getting response from Mistral:', error);
      setMessages(prev => [...prev, {
        text: `Sorry, there was an error getting a response: ${error.message}`,
        type: 'assistant'
      }]);
    } finally {
      setIsTyping(false);
    }
  }
};

  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      <Header />
      <div className="chat-container">
        <div className="model-status">
          <span className={`status-indicator ${llamaStatus.available ? 'online' : 'offline'}`}></span>
          <span className="status-text">
            {llamaStatus.available 
              ? `Mistral model is online` 
              : 'Mistral model is offline'}
          </span>
        </div>
        
        <div className="chat-messages">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.type}`}>
              <div className="message-avatar">
                {message.type === 'assistant' ? 'AI' : 'You'}
              </div>
              <div className="message-content">{message.text}</div>
            </div>
          ))}
          
          {isTyping && (
            <div className="message assistant">
              <div className="message-avatar">AI</div>
              <div className="message-content typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
        
        <div className="chat-input">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about the Echo Hall project or video analysis..."
            rows="2"
            disabled={isTyping}
          />
          <button 
            className="button" 
            onClick={sendMessage}
            disabled={isTyping || !input.trim()}
          >
            Send
          </button>
        </div>
      </div>
    </>
  );
}

export default ChatAgent;