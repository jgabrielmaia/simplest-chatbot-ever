import pytest
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, 'src')

from chatbot import chat


@pytest.fixture
def mock_chat_components():
    """Setup mocks for all chatbot components"""
    with patch('chatbot.OpenAI') as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        yield mock_client


class TestChatbot:
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_successful_conversation(self, mock_print, mock_input, mock_chat_components):
        """Test normal conversation flow with security checks"""
        mock_input.side_effect = ["Hello!", "exit"]
        
        # Mock API responses: exit(no), security(safe), chat, exit(yes)
        mock_chat_components.chat.completions.create.side_effect = [
            self._create_response("0"),  # exit check
            self._create_response("1"),  # security check
            self._create_response("Hi there!"),
            self._create_response("1")   # exit
        ]
        
        chat()
        
        assert mock_chat_components.chat.completions.create.call_count == 4
        assert self._printed(mock_print, "Hi there!")
        assert self._printed(mock_print, "Goodbye!")
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_immediate_exit(self, mock_print, mock_input, mock_chat_components):
        """Test user exits immediately"""
        mock_input.side_effect = ["bye"]
        mock_chat_components.chat.completions.create.return_value = self._create_response("1")
        
        chat()
        
        assert mock_chat_components.chat.completions.create.call_count == 1
        assert self._printed(mock_print, "Goodbye!")
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_multiple_interactions(self, mock_print, mock_input, mock_chat_components):
        """Test multiple messages before exit"""
        mock_input.side_effect = ["Hi", "Bye", "quit"]
        
        mock_chat_components.chat.completions.create.side_effect = [
            self._create_response("0"), self._create_response("1"), self._create_response("Hello!"),
            self._create_response("0"), self._create_response("1"), self._create_response("Goodbye!"),
            self._create_response("1")
        ]
        
        chat()
        
        assert mock_chat_components.chat.completions.create.call_count == 7
    
    @staticmethod
    def _create_response(content):
        response = MagicMock()
        response.choices[0].message.content = content
        return response
    
    @staticmethod
    def _printed(mock_print, text):
        return any(text in str(call) for call in mock_print.call_args_list)
