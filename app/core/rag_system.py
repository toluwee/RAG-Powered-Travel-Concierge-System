from typing import List, Dict, Any
import os
import logging
import json
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TravelRAGSystem:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self._initialize_rag()

    def _initialize_rag(self):
        """Initialize the RAG system with travel data"""
        # Create vector store directory if it doesn't exist
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        
        # Load and process travel data
        self._load_travel_data()
        
        logger.info("RAG system initialized (simplified version)")

    def _load_travel_data(self):
        """Load and process travel-related data"""
        # This would typically load from multiple sources
        # For now, we'll create some example data
        self.travel_data = [
            {
                "content": "Marriott hotels are preferred for business travelers in financial districts",
                "metadata": {"source": "corporate_policy", "type": "hotel"}
            },
            {
                "content": "Delta Airlines offers the best business class service for transatlantic flights",
                "metadata": {"source": "corporate_policy", "type": "flight"}
            },
            {
                "content": "Most business meetings in London are scheduled between 9 AM and 5 PM",
                "metadata": {"source": "travel_guide", "type": "meeting"}
            },
            {
                "content": "Airport transfers should be booked at least 24 hours in advance",
                "metadata": {"source": "travel_guide", "type": "transportation"}
            },
            {
                "content": "Business travelers prefer hotels with 24-hour business centers",
                "metadata": {"source": "preferences", "type": "hotel"}
            }
        ]

    def query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            # Simple keyword-based search for demo purposes
            relevant_docs = []
            query_lower = query.lower()
            
            for doc in self.travel_data:
                if any(keyword in doc["content"].lower() for keyword in query_lower.split()):
                    relevant_docs.append(doc)
            
            # Return mock response
            if relevant_docs:
                answer = f"Based on travel data: {relevant_docs[0]['content']}"
            else:
                answer = "Consider booking airport transfers in advance and ensure hotels have business centers."
            
            return {
                "answer": answer,
                "sources": relevant_docs[:top_k]  # Return up to top_k relevant sources
            }
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return {
                "error": str(e),
                "answer": "Consider booking airport transfers in advance.",
                "sources": []
            }

    def add_document(self, content: str, metadata: Dict[str, Any]):
        """Add a new document to the RAG system"""
        try:
            doc = {
                "content": content,
                "metadata": metadata
            }
            self.travel_data.append(doc)
            logger.info(f"Added document: {metadata.get('type', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return False

    def update_document(self, old_content: str, new_content: str, metadata: Dict[str, Any]):
        """Update an existing document in the RAG system"""
        # This is a simplified implementation
        self.add_document(new_content, metadata) 