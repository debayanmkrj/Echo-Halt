const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const readFileAsync = promisify(fs.readFile);
const readdirAsync = promisify(fs.readdir);
const statAsync = promisify(fs.stat);

class ProjectContextProvider {
  constructor(projectRoot) {
    this.projectRoot = projectRoot;
    this.cachedContext = null;
    this.lastUpdateTime = 0;
    this.cacheValidityPeriod = 5 * 60 * 1000; // 5 minutes in milliseconds
    
    // File extensions to include in context
    this.relevantExtensions = [
      '.js', '.jsx', '.ts', '.tsx', '.json', '.md', 
      '.py', '.html', '.css', '.scss', '.txt'
    ];
    
    // Files or directories to exclude
    this.excludePatterns = [
      'node_modules', 'venv', 'env', '.git', 
      'dist', 'build', 'public/videos', 'public/temp', 
      'public/original_videos', 'public/keyframes'
    ];

    // Maximum size of a single file to include
    this.maxFileSizeBytes = 100 * 1024; // 100KB
  }

  /**
   * Get project context as a string
   * @returns {Promise<string>}
   */
  async getProjectContext() {
    const now = Date.now();
    
    // Return cached context if it's recent enough
    if (this.cachedContext && (now - this.lastUpdateTime < this.cacheValidityPeriod)) {
      return this.cachedContext;
    }
    
    try {
      // Build fresh context
      const context = await this._buildProjectContext();
      
      // Cache the context
      this.cachedContext = context;
      this.lastUpdateTime = now;
      
      return context;
    } catch (error) {
      console.error('Error building project context:', error);
      return this.cachedContext || 'Error: Unable to build project context';
    }
  }
  
  /**
   * Build context from the project directory
   * @returns {Promise<string>}
   * @private
   */
  async _buildProjectContext() {
    // Start with basic project description
    let context = `Project: Echo Hall Video Analysis System
Project Root: ${this.projectRoot}
Description: A system for analyzing videos to detect manipulations and duplicates.

Key Components:
1. Frontend React Application
2. Node.js Express Server
3. Video Analysis Utilities

Project Structure Summary:
`;
    
    try {
      // Add project structure overview
      const structure = await this._getDirectoryStructure(this.projectRoot, 0, 2);
      context += structure;
      
      // Get key files content
      const keyFiles = [
        'server.js',
        'ChatAgent.js',
        'package.json',
        'video_analysis_utils.js'
      ];
      
      // Add content of key files
      context += '\n\nKey File Contents:\n';
      
      for (const file of keyFiles) {
        try {
          // Find file path (could be in subdirectories)
          const filePath = await this._findFile(this.projectRoot, file);
          
          if (filePath) {
            const stats = await statAsync(filePath);
            
            // Check file size
            if (stats.size <= this.maxFileSizeBytes) {
              const content = await readFileAsync(filePath, 'utf8');
              context += `\n--- ${file} ---\n${content}\n`;
            } else {
              context += `\n--- ${file} ---\n(File too large to include in context)\n`;
            }
          }
        } catch (error) {
          console.error(`Error reading ${file}:`, error);
        }
      }
      
      // Add brief info about the video analysis capabilities
      context += `
\nVideo Analysis Capabilities:
- Perceptual hashing for similarity detection
- Histogram comparison for color analysis
- Shot boundary detection for scene changes
- Keyframe extraction for content analysis
- Object detection for identifying common elements
- Visual embedding for deep feature analysis
- Split screen detection
- Chroma key detection (green screen)
`;
      
      return context;
    } catch (error) {
      console.error('Error building project context:', error);
      return 'Error: Unable to build project context';
    }
  }
  
  /**
   * Find a file in the project directory tree
   * @param {string} dir - Directory to search in
   * @param {string} filename - File to find
   * @returns {Promise<string|null>} - Full path to the file if found
   * @private
   */
  async _findFile(dir, filename) {
    try {
      const files = await readdirAsync(dir);
      
      for (const file of files) {
        // Skip excluded patterns
        if (this.excludePatterns.some(pattern => file.includes(pattern))) {
          continue;
        }
        
        const filePath = path.join(dir, file);
        const stats = await statAsync(filePath);
        
        if (stats.isDirectory()) {
          // Recursively search subdirectories
          const foundPath = await this._findFile(filePath, filename);
          if (foundPath) return foundPath;
        } else if (file === filename) {
          return filePath;
        }
      }
    } catch (error) {
      console.error(`Error searching for ${filename} in ${dir}:`, error);
    }
    
    return null;
  }
  
  /**
   * Get simplified directory structure
   * @param {string} dir - Directory to process
   * @param {number} level - Current recursion level
   * @param {number} maxDepth - Maximum recursion depth
   * @returns {Promise<string>}
   * @private
   */
  async _getDirectoryStructure(dir, level, maxDepth) {
    if (level > maxDepth) return '';
    
    try {
      const files = await readdirAsync(dir);
      let structure = '';
      
      for (const file of files) {
        // Skip excluded patterns
        if (this.excludePatterns.some(pattern => file.includes(pattern))) {
          continue;
        }
        
        const filePath = path.join(dir, file);
        const stats = await statAsync(filePath);
        const indent = '  '.repeat(level);
        
        if (stats.isDirectory()) {
          structure += `${indent}- ${file}/\n`;
          const subStructure = await this._getDirectoryStructure(filePath, level + 1, maxDepth);
          structure += subStructure;
        } else {
          const ext = path.extname(file);
          if (this.relevantExtensions.includes(ext)) {
            structure += `${indent}- ${file}\n`;
          }
        }
      }
      
      return structure;
    } catch (error) {
      console.error(`Error getting structure for ${dir}:`, error);
      return '';
    }
  }
  
  /**
   * Check if a query is related to the project
   * @param {string} query - User query
   * @returns {boolean}
   */
  isRelevantQuery(query) {
    const projectTerms = [
      'video', 'analysis', 'echo hall', 'manipulation', 'detection',
      'similarity', 'compare', 'hash', 'histogram', 'keyframe',
      'object', 'shot', 'boundary', 'react', 'express', 'server',
      'upload', 'similar', 'unique', 'project', 'folder', 'file',
      'ollama', 'llama', 'ai', 'model', 'integration'
    ];
    
    const queryLower = query.toLowerCase();
    
    // Check for presence of project-related terms
    return projectTerms.some(term => queryLower.includes(term));
  }
}

module.exports = ProjectContextProvider;