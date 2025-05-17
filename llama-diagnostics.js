const axios = require('axios');
const fs = require('fs');
const path = require('path');

async function testModelAvailability() {
    console.log('1. Model Availability Check:');
    try {
        const response = await axios.get('http://localhost:11434/api/tags');
        const models = response.data.models || [];
        console.log('   Available Models:');
        models.forEach(model => {
            console.log(`   - ${model.name}`);
        });
        return models;
    } catch (error) {
        console.error('   Failed to retrieve models:', error.message);
        return [];
    }
}

async function testSimpleGeneration() {
    console.log('\n2. Simple Generation Test:');
    try {
        const response = await axios.post('http://localhost:11434/api/generate', {
            model: 'llama2',
            prompt: 'Hello, world!',
            stream: false,
            options: {
                // Reduce complexity to minimize potential issues
                temperature: 0.1,
                num_predict: 10
            }
        }, {
            headers: {
                'Content-Type': 'application/json'
            },
            timeout: 30000  // Increased timeout
        });

        console.log('   Generation Successful');
        console.log('   Response Preview:', 
            response.data.response ? 
            response.data.response.substring(0, 200) : 
            'No response text'
        );
        return true;
    } catch (error) {
        console.error('   Generation Failed:', error.message);
        
        // Detailed error logging
        if (error.response) {
            console.error('   Response Data:', error.response.data);
            console.error('   Response Status:', error.response.status);
        } else if (error.request) {
            console.error('   No response received. Request was made but no response.');
        } else {
            console.error('   Error setting up the request:', error.message);
        }
        return false;
    }
}

async function testStreamingGeneration() {
    console.log('\n3. Streaming Generation Test:');
    try {
        const response = await axios.post('http://localhost:11434/api/generate', {
            model: 'llama2',
            prompt: 'Explain the concept of AI in one paragraph.',
            stream: true
        }, {
            headers: {
                'Content-Type': 'application/json'
            },
            responseType: 'stream',
            timeout: 30000
        });

        return new Promise((resolve, reject) => {
            let fullResponse = '';
            let chunkCount = 0;

            response.data.on('data', (chunk) => {
                try {
                    const lines = chunk.toString().split('\n').filter(line => line.trim());
                    lines.forEach(line => {
                        try {
                            const data = JSON.parse(line);
                            if (data.response) {
                                fullResponse += data.response;
                                chunkCount++;
                            }
                        } catch (parseError) {
                            console.error('   Parsing error in chunk:', parseError);
                        }
                    });
                } catch (error) {
                    console.error('   Error processing chunk:', error);
                }
            });

            response.data.on('end', () => {
                console.log('   Streaming Completed');
                console.log(`   Total Chunks: ${chunkCount}`);
                console.log('   Response Preview:', 
                    fullResponse ? fullResponse.substring(0, 200) + '...' : 'No response'
                );
                resolve(true);
            });

            response.data.on('error', (error) => {
                console.error('   Streaming Error:', error);
                reject(false);
            });
        });
    } catch (error) {
        console.error('   Streaming Generation Failed:', error.message);
        return false;
    }
}

async function runOllamaDiagnostics() {
    console.log('Ollama Comprehensive Diagnostics\n');
    
    // Run tests sequentially
    const models = await testModelAvailability();
    const simpleGen = await testSimpleGeneration();
    const streamingGen = await testStreamingGeneration();
    
    console.log('\nDiagnostic Summary:');
    console.log('Models Available:', models.length > 0 ? 'YES ✓' : 'NO ✗');
    console.log('Simple Generation:', simpleGen ? 'PASS ✓' : 'FAIL ✗');
    console.log('Streaming Generation:', streamingGen ? 'PASS ✓' : 'FAIL ✗');
}

runOllamaDiagnostics();