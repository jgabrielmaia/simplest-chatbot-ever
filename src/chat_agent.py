from openai import OpenAI


class ChatAgent:
    """Agent responsible for conversational interactions"""
    
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.conversation_history = []
    
    def respond(self, user_input: str) -> str:
        """Generate a response to user input"""
        self.conversation_history.append({"role": "user", "content": user_input})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history
        )
        
        bot_message = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": bot_message})
        
        return bot_message
