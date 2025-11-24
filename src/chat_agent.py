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
    
    def respond_stream(self, user_input: str):
        """Generate a streaming response to user input"""
        self.conversation_history.append({"role": "user", "content": user_input})
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            stream=True
        )
        
        bot_message = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                bot_message += content
                yield content
        
        self.conversation_history.append({"role": "assistant", "content": bot_message})
