import chromadb
from openai import OpenAI
from typing import List, Dict
from datetime import datetime


class MemoryStore:
    """Stores and retrieves conversation history using ChromaDB"""
    
    def __init__(self, client: OpenAI, persist_dir: str = "./chroma_data", collection_name: str = "conversations"):
        self.client = client
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
    
    def store_turn(self, session_id: str, turn_number: int, user_message: str, assistant_message: str):
        """Store a conversation turn (user + assistant pair)"""
        document = f"User: {user_message}\nAssistant: {assistant_message}"
        doc_id = f"{session_id}_turn{turn_number}"
        
        embedding = self._generate_embedding(user_message)
        
        self.collection.add(
            documents=[document],
            embeddings=[embedding],
            metadatas=[{
                "session_id": session_id,
                "turn_number": turn_number,
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message
            }],
            ids=[doc_id]
        )
    
    def retrieve_relevant(self, query: str, top_k: int = 3, session_id: str = None) -> List[Dict[str, str]]:
        """Retrieve top-k most relevant conversation turns (optionally filtered by session)"""
        if self.collection.count() == 0:
            return []
        
        query_embedding = self._generate_embedding(query)
        
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count())
        }
        
        if session_id:
            query_params["where"] = {"session_id": session_id}
        
        results = self.collection.query(**query_params)
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        relevant_turns = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            relevant_turns.append({
                "content": doc,
                "turn_number": metadata['turn_number'],
                "timestamp": metadata['timestamp']
            })
        
        return relevant_turns
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
