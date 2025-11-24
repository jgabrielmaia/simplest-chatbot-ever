import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    return MagicMock()


@pytest.fixture
def mock_response():
    """Factory for creating mock API responses"""
    def _create_response(content: str):
        response = MagicMock()
        response.choices[0].message.content = content
        return response
    return _create_response
