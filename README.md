# Simple Chatbot

A minimal chatbot using Python and OpenAI API with security features.

## Features

-  Intent-based exit detection
-  Security layer to prevent prompt injection
-  Conversation memory
-  Clean agent architecture
-  Comprehensive test coverage

## Project Structure

```
chatbot/
 src/
    chatbot.py           # Main orchestration
    chat_agent.py        # Conversation handler
    intent_classifier.py # Binary intent classifier
 tests/
    conftest.py          # Test fixtures
    test_chatbot.py      # Integration tests
    test_intent_classifier.py  # Unit tests
    test_security.py     # Security tests
 main.py                  # Entry point
 .env                     # API keys (not in git)
 .env.example             # Example config
 requirements.txt         # Dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o
```

3. Run the chatbot:
```bash
python main.py
```

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_security.py -v
```

## Usage

Type your messages and press Enter. Say goodbye naturally to exit (e.g., "bye", "see you later", "quit").
