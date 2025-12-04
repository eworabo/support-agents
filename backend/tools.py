"""
Fixed tools.py - Uses Supabase pgvector directly (no FAISS)
"""
from typing import Type, Any, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
from langchain_openai import OpenAIEmbeddings
from supabase import create_client

# Initialize Supabase
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

class SearchKBInput(BaseModel):
    """Input schema for SearchKBTool."""
    query: Optional[str] = Field(
        None, 
        description="The search query string to find relevant knowledge base entries. Example: 'password reset' or 'refund policy'"
    )

class SearchKBTool(BaseTool):
    name: str = "search_kb"
    description: str = (
        "Search the knowledge base using a query string. "
        "Returns relevant KB entries with confidence scores. "
        "Usage: search_kb(query='your search terms here')"
    )
    args_schema: Type[BaseModel] = SearchKBInput

    def _run(self, query: str = None, **kwargs) -> str:
        """
        Searches the knowledge base using Supabase pgvector.
        NO FAISS - queries database directly for always-fresh results.
        """
        # Handle parameter variations
        if query is None:
            query = kwargs.get('description') or kwargs.get('search_query') or kwargs.get('q')
            if query is None:
                return "Error: No search query provided. Please provide a 'query' parameter."
        
        try:
            # Generate query embedding
            query_embedding = embeddings.embed_query(query)
            
            # Search using Supabase pgvector RPC function
            # This queries the database directly - always fresh!
            result = supabase.rpc('match_kb_entries', {
                'query_embedding': query_embedding,
                'match_threshold': 0.7,  # Minimum similarity score
                'match_count': 3  # Top 3 results
            }).execute()
            
            if not result.data:
                return "No relevant matches found in knowledge base."
            
            # Format results
            output = []
            for match in result.data:
                similarity = match.get('similarity', 0)
                title = match.get('title', 'Untitled')
                content = match.get('content', '')
                file_url = match.get('file_url', '')
                
                # Determine confidence level
                confidence = similarity
                if confidence >= 0.82:
                    confidence_label = "High-confidence"
                    emoji = "✅"
                else:
                    confidence_label = "Low-confidence"
                    emoji = "⚠️"
                
                # Format entry
                entry_text = f"{emoji} {confidence_label} match ({confidence:.2f}):\n"
                entry_text += f"Title: {title}\n"
                entry_text += f"Content: {content}\n"
                
                # Include file URLs if present
                if file_url:
                    entry_text += f"Attachments: {file_url}\n"
                
                output.append(entry_text)
            
            return "\n" + ("-" * 50) + "\n\n".join(output)
            
        except Exception as e:
            error_msg = f"Error searching KB: {str(e)}"
            print(error_msg)
            return error_msg

# Create instance
search_kb = SearchKBTool()