from openai import OpenAI
from src.chat_agent import ChatAgent
from src.memory_store import MemoryStore
from typing import Optional


class RAGChatAgent:
    """Chat agent with RAG-based conversation memory"""
    
    def __init__(self, client: OpenAI, model: str, session_id: str, 
                 memory_store: Optional[MemoryStore] = None, 
                 top_k: int = 3, recent_turns: int = 2):
        self.client = client
        self.model = model
        self.session_id = session_id
        self.top_k = top_k
        self.recent_turns = recent_turns
        self.turn_counter = 0
        
        self.chat_agent = ChatAgent(client, model)
        self.memory_store = memory_store or MemoryStore(client)
    
    def respond(self, user_input: str) -> str:
        """Generate response with memory-augmented context"""
        relevant_memories = self.memory_store.retrieve_relevant(
            user_input, 
            self.top_k
        )
        
        augmented_history = self._build_context(relevant_memories)
        
        original_history = self.chat_agent.conversation_history
        self.chat_agent.conversation_history = augmented_history
        
        response = self.chat_agent.respond(user_input)
        
        self.chat_agent.conversation_history = original_history
        self.turn_counter += 1
        self.memory_store.store_turn(self.session_id, self.turn_counter, user_input, response)
        
        original_history.append({"role": "user", "content": user_input})
        original_history.append({"role": "assistant", "content": response})
        
        return response
    
    def respond_stream(self, user_input: str):
        """Generate streaming response with memory-augmented context"""
        relevant_memories = self.memory_store.retrieve_relevant(
            user_input,
            self.top_k
        )
        
        augmented_history = self._build_context(relevant_memories)
        
        original_history = self.chat_agent.conversation_history
        self.chat_agent.conversation_history = augmented_history
        
        full_response = ""
        for chunk in self.chat_agent.respond_stream(user_input):
            full_response += chunk
            yield chunk
        
        self.chat_agent.conversation_history = original_history
        self.turn_counter += 1
        self.memory_store.store_turn(self.session_id, self.turn_counter, user_input, full_response)
        
        original_history.append({"role": "user", "content": user_input})
        original_history.append({"role": "assistant", "content": full_response})
    
    def _build_context(self, relevant_memories: list) -> list:
        """Combine retrieved memories with recent conversation history"""
        context = []
        
        if relevant_memories:
            memory_text = "Previous relevant conversations:\n\n"
            for memory in relevant_memories:
                memory_text += f"{memory['content']}\n\n"
            context.append({"role": "system", "content": memory_text})
        
        recent_history = self.chat_agent.conversation_history[-self.recent_turns*2:] if self.chat_agent.conversation_history else []
        context.extend(recent_history)
        
        return context
