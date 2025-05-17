const axios = require('axios');

class GPT4AllClient {
  constructor(baseURL = 'http://localhost:3004') {
    this.baseURL = baseURL;
    this.axios = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  /**
   * Generate a completion using GPT4All
   * @param {string} prompt - The user's prompt
   * @param {object} options - Additional options for the API
   * @returns {Promise<string>} - The model's response
   */
  async generateCompletion(prompt, options = {}) {
    try {
      const response = await this.axios.post('/api/chat', {
        prompt,
        ...options
      });
      
      return response.data.response;
    } catch (error) {
      console.error('GPT4All API error:', error.message);
      throw new Error(`Failed to generate completion: ${error.message}`);
    }
  }

  /**
   * Check if GPT4All service is available
   * @returns {Promise<boolean>}
   */
  async isAvailable() {
    try {
      await this.axios.get('/api/status');
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * List available models (placeholder for compatibility)
   * @returns {Promise<Array>}
   */
  async listModels() {
    try {
      const response = await this.axios.get('/api/models');
      return response.data.models || ['mistral-7b-instruct'];
    } catch (error) {
      console.error('Failed to list models:', error.message);
      return ['mistral-7b-instruct'];
    }
  }
}

module.exports = GPT4AllClient;