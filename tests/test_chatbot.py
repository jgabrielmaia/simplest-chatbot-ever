import pytest
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, 'src')

from chatbot import chat


@pytest.fixture
def mock_chat_components():
    """Setup mocks for all chatbot components"""
    with patch('chatbot.OpenAI') as mock_openai_class, \
         patch('chatbot.Console') as mock_console_class, \
         patch('chatbot.Prompt') as mock_prompt_class:
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        yield {'client': mock_client, 'console': mock_console, 'prompt': mock_prompt_class}


class TestChatbot:
    
    def test_successful_conversation(self, mock_chat_components):
        """Test normal conversation flow with security checks"""
        mock_chat_components['prompt'].ask.side_effect = ["Hello!", "exit"]
        
        # Mock API responses: exit(no), security(safe), streaming chat, exit(yes)
        mock_chat_components['client'].chat.completions.create.side_effect = [
            self._create_response("0"),  # exit check
            self._create_response("1"),  # security check
            self._create_stream(["Hi ", "there!"]),  # streaming chat
            self._create_response("1")   # exit
        ]
        
        chat()
        
        assert mock_chat_components['client'].chat.completions.create.call_count == 4
        console_calls = [str(call) for call in mock_chat_components['console'].print.call_args_list]
        assert any("Goodbye!" in call for call in console_calls)
    
    def test_immediate_exit(self, mock_chat_components):
        """Test user exits immediately"""
        mock_chat_components['prompt'].ask.side_effect = ["bye"]
        mock_chat_components['client'].chat.completions.create.return_value = self._create_response("1")
        
        chat()
        
        assert mock_chat_components['client'].chat.completions.create.call_count == 1
        console_calls = [str(call) for call in mock_chat_components['console'].print.call_args_list]
        assert any("Goodbye!" in call for call in console_calls)
    
    def test_multiple_interactions(self, mock_chat_components):
        """Test multiple messages before exit"""
        mock_chat_components['prompt'].ask.side_effect = ["Hi", "Bye", "quit"]
        
        mock_chat_components['client'].chat.completions.create.side_effect = [
            self._create_response("0"), self._create_response("1"), self._create_stream(["Hello!"]),
            self._create_response("0"), self._create_response("1"), self._create_stream(["Goodbye!"]),
            self._create_response("1")
        ]
        
        chat()
        
        assert mock_chat_components['client'].chat.completions.create.call_count == 7
    
    @staticmethod
    def _create_response(content):
        response = MagicMock()
        response.choices[0].message.content = content
        return response
    
    @staticmethod
    def _create_stream(chunks):
        """Create a mock streaming response"""
        stream_chunks = []
        for chunk_text in chunks:
            chunk = MagicMock()
            chunk.choices[0].delta.content = chunk_text
            stream_chunks.append(chunk)
        return iter(stream_chunks)
