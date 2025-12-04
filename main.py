"""
Simple Chatbot Entry Point

Usage:
  python main.py              # Run without memory
  python main.py --memory     # Run with RAG memory
  python main.py --inspect    # Inspect memory store
"""
import sys
from src.chatbot import chat
from inspect_memory import inspect_memory

if __name__ == "__main__":
    if "--inspect" in sys.argv:
        inspect_memory()
    else:
        use_memory = "--memory" in sys.argv
        chat(use_memory=use_memory)
