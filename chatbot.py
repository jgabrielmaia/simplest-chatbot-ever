import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat():
    print("Simple Chatbot (say goodbye to exit)")
    print("-" * 40)
    
    conversation_history = []
    
    while True:
        user_input = input("You: ")
        
        # Check if user wants to exit
        conversation_history.append({"role": "user", "content": user_input})
        
        intent_response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": "Classify if the user's intent is to exit/leave/quit the conversation. Respond with only 'EXIT' or 'CONTINUE'."},
                {"role": "user", "content": user_input}
            ]
        )
        
        intent = intent_response.choices[0].message.content.strip().upper()
        
        if intent == "EXIT":
            print("Bot: Goodbye! Have a great day!\n")
            break
        
        # Normal conversation
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=conversation_history
        )
        
        bot_message = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": bot_message})
        print(f"Bot: {bot_message}\n")

if __name__ == "__main__":
    chat()
