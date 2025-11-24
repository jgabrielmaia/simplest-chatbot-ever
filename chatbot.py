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
        
        # Check if user wants to exit using logit_bias
        intent_response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": "Classify the user's intent. Respond with only '0' if they want to continue the conversation, or '1' if they want to exit/leave/quit."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=1,
            temperature=0,
            logit_bias={
                "15": 100,  # Token for "0"
                "16": 100   # Token for "1"
            }
        )
        
        intent = intent_response.choices[0].message.content.strip()
        
        if intent == "1":
            print("Bot: Goodbye! Have a great day!\n")
            break
        
        # Add to conversation history only if continuing
        conversation_history.append({"role": "user", "content": user_input})
        
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
