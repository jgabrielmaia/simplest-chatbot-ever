import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat():
    print("Simple Chatbot (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "user", "content": user_input}
            ]
        )
        
        bot_message = response.choices[0].message.content
        print(f"Bot: {bot_message}\n")

if __name__ == "__main__":
    chat()
