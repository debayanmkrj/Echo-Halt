from gpt4all import GPT4All
import json

class GPT4AllChatAgent:
    def __init__(self, model_name='mistral-7b-instruct-v0.1.Q4_0.gguf'):
        """
        Initialize GPT4All model
        
        Args:
            model_name (str): Name of the model to load
        """
        try:
            # Load the model
            self.model = GPT4All(model_name)
        except Exception as e:
            print(f"Model loading error: {e}")
            raise

    def generate_response(self, prompt, max_tokens=500):
        """
        Generate a response to a given prompt
        
        Args:
            prompt (str): Input text to generate response for
            max_tokens (int): Maximum length of generated response
        
        Returns:
            dict: Response dictionary with status and text
        """
        try:
            # Start a chat session
            with self.model.chat_session():
                # Generate response
                response = self.model.generate(
                    prompt, 
                    max_tokens=max_tokens, 
                    temp=0.7  # Creativity/randomness
                )
                
                return {
                    'status': 'success',
                    'response': response
                }
        except Exception as e:
            print(f"Response generation error: {e}")
            return {
                'status': 'error',
                'response': f"An error occurred: {str(e)}"
            }

    def process_query(self, query, project_context):
        """
        Process a query with project context
        
        Args:
            query (str): User's query
            project_context (str): Context about the project
        
        Returns:
            dict: Processed response
        """
        # Construct a detailed prompt
        full_prompt = f"""You are an AI assistant for the Echo Hall video analysis project.
Project Context:
{project_context}

User Query: {query}

Provide a helpful, accurate response focusing on the project's capabilities and context."""

        return self.generate_response(full_prompt)

# Example usage for testing
if __name__ == '__main__':
    agent = GPT4AllChatAgent()
    test_query = "Tell me about the video analysis capabilities of this project"
    result = agent.generate_response(test_query)
    print(json.dumps(result, indent=2))