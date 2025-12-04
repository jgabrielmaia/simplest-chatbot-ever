import pytest
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, 'src')

from memory_store import MemoryStore


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    client = MagicMock()
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
    client.embeddings.create.return_value = mock_embedding_response
    return client


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client"""
    with patch('memory_store.chromadb.PersistentClient') as mock_chroma:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        yield mock_collection


class TestMemoryStore:
    
    def test_store_conversation_turn(self, mock_openai_client, mock_chroma_client):
        """Test storing a conversation turn"""
        store = MemoryStore(mock_openai_client)
        
        store.store_turn("session123", 1, "Hello", "Hi there!")
        
        assert mock_chroma_client.add.called
        call_args = mock_chroma_client.add.call_args[1]
        assert "session123" in call_args['ids'][0]
        assert "Hello" in call_args['documents'][0]
        assert "Hi there!" in call_args['documents'][0]
        assert call_args['metadatas'][0]['session_id'] == "session123"
        assert call_args['metadatas'][0]['turn_number'] == 1
    
    def test_retrieve_relevant_memories(self, mock_openai_client, mock_chroma_client):
        """Test retrieving relevant conversation turns across all sessions"""
        mock_chroma_client.count.return_value = 2
        mock_chroma_client.query.return_value = {
            'documents': [["User: test\nAssistant: response"]],
            'metadatas': [[{'turn_number': 1, 'timestamp': '2025-01-01T00:00:00'}]]
        }
        
        store = MemoryStore(mock_openai_client)
        results = store.retrieve_relevant("test query", top_k=3)
        
        assert len(results) == 1
        assert results[0]['content'] == "User: test\nAssistant: response"
        assert results[0]['turn_number'] == 1
        assert mock_chroma_client.query.called
        
        call_args = mock_chroma_client.query.call_args[1]
        assert 'where' not in call_args
    
    def test_retrieve_from_empty_store(self, mock_openai_client, mock_chroma_client):
        """Test retrieval from empty memory store"""
        mock_chroma_client.count.return_value = 0
        
        store = MemoryStore(mock_openai_client)
        results = store.retrieve_relevant("test query")
        
        assert results == []
        assert not mock_chroma_client.query.called
    
    def test_session_isolation(self, mock_openai_client, mock_chroma_client):
        """Test that retrieval filters by session_id when provided"""
        mock_chroma_client.count.return_value = 5
        mock_chroma_client.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        store = MemoryStore(mock_openai_client)
        store.retrieve_relevant("query", top_k=3, session_id="session123")
        
        call_args = mock_chroma_client.query.call_args[1]
        assert call_args['where'] == {"session_id": "session123"}
    
    @pytest.mark.parametrize("text_input", [
        "Hello world",
        "What is the weather?",
        "Tell me about AI"
    ])
    def test_embedding_generation(self, mock_openai_client, mock_chroma_client, text_input):
        """Test embedding API is called correctly"""
        store = MemoryStore(mock_openai_client)
        embedding = store._generate_embedding(text_input)
        
        assert mock_openai_client.embeddings.create.called
        call_args = mock_openai_client.embeddings.create.call_args[1]
        assert call_args['model'] == "text-embedding-3-small"
        assert call_args['input'] == text_input
        assert len(embedding) == 1536
