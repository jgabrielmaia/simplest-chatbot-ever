import pytest
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, 'src')

from rag_chat_agent import RAGChatAgent


@pytest.fixture
def mock_dependencies():
    """Mock all RAGChatAgent dependencies"""
    with patch('rag_chat_agent.ChatAgent') as mock_chat_agent_class, \
         patch('rag_chat_agent.MemoryStore') as mock_memory_store_class:
        
        mock_client = MagicMock()
        mock_chat_agent = MagicMock()
        mock_chat_agent.conversation_history = []
        mock_chat_agent_class.return_value = mock_chat_agent
        
        mock_memory_store = MagicMock()
        mock_memory_store_class.return_value = mock_memory_store
        
        yield {
            'client': mock_client,
            'chat_agent': mock_chat_agent,
            'memory_store': mock_memory_store
        }


class TestRAGChatAgent:
    
    def test_respond_with_memory_context(self, mock_dependencies):
        """Test that relevant memories are retrieved and used"""
        mock_dependencies['memory_store'].retrieve_relevant.return_value = [
            {"content": "User: topic X\nAssistant: about X", "turn_number": 1, "timestamp": "2025-01-01"}
        ]
        mock_dependencies['chat_agent'].respond.return_value = "Response with context"
        
        agent = RAGChatAgent(
            mock_dependencies['client'],
            "gpt-4o-mini",
            "session123",
            memory_store=mock_dependencies['memory_store']
        )
        
        response = agent.respond("Tell me more about X")
        
        assert mock_dependencies['memory_store'].retrieve_relevant.called
        retrieval_args = mock_dependencies['memory_store'].retrieve_relevant.call_args[0]
        assert retrieval_args[0] == "Tell me more about X"
        
        assert mock_dependencies['memory_store'].store_turn.called
        assert response == "Response with context"
    
    def test_recent_context_prioritized(self, mock_dependencies):
        """Test that recent conversation turns are included"""
        mock_dependencies['chat_agent'].conversation_history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": "Second response"}
        ]
        mock_dependencies['memory_store'].retrieve_relevant.return_value = []
        mock_dependencies['chat_agent'].respond.return_value = "New response"
        
        agent = RAGChatAgent(
            mock_dependencies['client'],
            "gpt-4o-mini",
            "session123",
            memory_store=mock_dependencies['memory_store'],
            recent_turns=2
        )
        
        agent.respond("Third message")
        
        assert mock_dependencies['chat_agent'].respond.called
    
    def test_fallback_without_relevant_memories(self, mock_dependencies):
        """Test agent works normally when no relevant memories found"""
        mock_dependencies['memory_store'].retrieve_relevant.return_value = []
        mock_dependencies['chat_agent'].respond.return_value = "Normal response"
        
        agent = RAGChatAgent(
            mock_dependencies['client'],
            "gpt-4o-mini",
            "session123",
            memory_store=mock_dependencies['memory_store']
        )
        
        response = agent.respond("New topic")
        
        assert response == "Normal response"
        assert mock_dependencies['memory_store'].store_turn.called
    
    def test_respond_stream_with_memory(self, mock_dependencies):
        """Test streaming response with memory context"""
        mock_dependencies['memory_store'].retrieve_relevant.return_value = [
            {"content": "User: X\nAssistant: Y", "turn_number": 1, "timestamp": "2025-01-01"}
        ]
        mock_dependencies['chat_agent'].respond_stream.return_value = iter(["Hello", " ", "world"])
        
        agent = RAGChatAgent(
            mock_dependencies['client'],
            "gpt-4o-mini",
            "session123",
            memory_store=mock_dependencies['memory_store']
        )
        
        chunks = list(agent.respond_stream("Test query"))
        
        assert chunks == ["Hello", " ", "world"]
        assert mock_dependencies['memory_store'].retrieve_relevant.called
        assert mock_dependencies['memory_store'].store_turn.called
        
        store_args = mock_dependencies['memory_store'].store_turn.call_args[0]
        assert store_args[3] == "Hello world"
    
    def test_turn_counter_increments(self, mock_dependencies):
        """Test that turn counter increments correctly"""
        mock_dependencies['memory_store'].retrieve_relevant.return_value = []
        mock_dependencies['chat_agent'].respond.return_value = "Response"
        
        agent = RAGChatAgent(
            mock_dependencies['client'],
            "gpt-4o-mini",
            "session123",
            memory_store=mock_dependencies['memory_store']
        )
        
        agent.respond("First")
        agent.respond("Second")
        agent.respond("Third")
        
        calls = mock_dependencies['memory_store'].store_turn.call_args_list
        assert len(calls) == 3
        assert calls[0][0][1] == 1
        assert calls[1][0][1] == 2
        assert calls[2][0][1] == 3
