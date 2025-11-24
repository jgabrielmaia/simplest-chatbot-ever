import pytest
from unittest.mock import patch, MagicMock
from chatbot import chat


class TestChatbot:
    
    @patch('chatbot.client')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_chat_successful_interaction(self, mock_print, mock_input, mock_client):
        """Test a successful chat interaction with the bot"""
        # Setup mock responses
        mock_input.side_effect = ["Hello!", "exit"]
        
        # Mock the intent classifier (first call returns "0" for continue, second returns "1" for exit)
        intent_response_continue = MagicMock()
        intent_response_continue.choices[0].message.content = "0"
        
        intent_response_exit = MagicMock()
        intent_response_exit.choices[0].message.content = "1"
        
        # Mock the actual chat response
        chat_response = MagicMock()
        chat_response.choices[0].message.content = "Hi there! How can I help you?"
        
        mock_client.chat.completions.create.side_effect = [
            intent_response_continue,  # First intent check
            chat_response,              # Chat response
            intent_response_exit        # Second intent check (exit)
        ]
        
        # Run the chat function
        chat()
        
        # Verify the correct number of API calls
        assert mock_client.chat.completions.create.call_count == 3
        
        # Verify the chat response was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Hi there! How can I help you?" in call for call in print_calls)
        assert any("Goodbye! Have a great day!" in call for call in print_calls)
    
    @patch('chatbot.client')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_chat_immediate_exit(self, mock_print, mock_input, mock_client):
        """Test exiting the chat immediately"""
        # User says "bye" right away
        mock_input.side_effect = ["bye"]
        
        # Mock the intent classifier returning "1" (exit)
        intent_response_exit = MagicMock()
        intent_response_exit.choices[0].message.content = "1"
        
        mock_client.chat.completions.create.return_value = intent_response_exit
        
        # Run the chat function
        chat()
        
        # Verify only one API call (intent check)
        assert mock_client.chat.completions.create.call_count == 1
        
        # Verify goodbye message was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Goodbye! Have a great day!" in call for call in print_calls)
    
    @patch('chatbot.client')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_chat_multiple_interactions_before_exit(self, mock_print, mock_input, mock_client):
        """Test multiple chat interactions before exiting"""
        # User has a conversation then exits
        mock_input.side_effect = ["What's the weather?", "Tell me a joke", "goodbye"]
        
        # Mock responses
        intent_continue = MagicMock()
        intent_continue.choices[0].message.content = "0"
        
        intent_exit = MagicMock()
        intent_exit.choices[0].message.content = "1"
        
        weather_response = MagicMock()
        weather_response.choices[0].message.content = "I don't have real-time weather data."
        
        joke_response = MagicMock()
        joke_response.choices[0].message.content = "Why did the chicken cross the road?"
        
        mock_client.chat.completions.create.side_effect = [
            intent_continue,    # First intent check
            weather_response,   # First chat response
            intent_continue,    # Second intent check
            joke_response,      # Second chat response
            intent_exit         # Third intent check (exit)
        ]
        
        # Run the chat function
        chat()
        
        # Verify the correct number of API calls
        assert mock_client.chat.completions.create.call_count == 5
        
        # Verify both responses were printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("I don't have real-time weather data." in call for call in print_calls)
        assert any("Why did the chicken cross the road?" in call for call in print_calls)
        assert any("Goodbye! Have a great day!" in call for call in print_calls)
