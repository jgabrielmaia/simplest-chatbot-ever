import pytest
from unittest.mock import MagicMock
import sys
sys.path.insert(0, 'src')

from intent_classifier import IntentClassifier


class TestIntentClassifier:
    
    @pytest.fixture
    def classifier(self, mock_openai_client):
        return IntentClassifier(mock_openai_client, "gpt-4o", "Test prompt")
    
    def test_classify_returns_binary_values(self, classifier, mock_openai_client, mock_response):
        """Test classify returns '0' or '1'"""
        mock_openai_client.chat.completions.create.side_effect = [
            mock_response("0"),
            mock_response("1")
        ]
        
        assert classifier.classify("input1") == "0"
        assert classifier.classify("input2") == "1"
    
    def test_is_positive(self, classifier, mock_openai_client, mock_response):
        """Test is_positive interprets '1' as True, '0' as False"""
        mock_openai_client.chat.completions.create.side_effect = [
            mock_response("1"),
            mock_response("0")
        ]
        
        assert classifier.is_positive("input1") is True
        assert classifier.is_positive("input2") is False
    
    def test_whitespace_handling(self, classifier, mock_openai_client, mock_response):
        """Test that responses are stripped of whitespace"""
        mock_openai_client.chat.completions.create.return_value = mock_response("  1  \n")
        
        assert classifier.classify("input") == "1"
    
    def test_api_parameters(self, classifier, mock_openai_client, mock_response):
        """Test correct API parameters are used"""
        mock_openai_client.chat.completions.create.return_value = mock_response("1")
        
        classifier.classify("test input")
        
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs['model'] == "gpt-4o"
        assert call_kwargs['max_tokens'] == 1
        assert call_kwargs['temperature'] == 0
        assert call_kwargs['logit_bias'] == {"15": 100, "16": 100}
        
        messages = call_kwargs['messages']
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == 'test input'
