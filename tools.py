from crewai.tools import BaseTool
from kb_vectorstore import vectorstore
from typing import Type
from pydantic import BaseModel, Field


class SearchKBInput(BaseModel):
    """Input for SearchKB tool."""
    query: str = Field(..., description="Search query to find relevant KB entries")


class SearchKBTool(BaseTool):
    name: str = "Search Knowledge Base"
    description: str = "Search the knowledge base for relevant support information. Returns KB entries with relevance scores (0-1 cosine similarity)."
    args_schema: Type[BaseModel] = SearchKBInput
    
    def _run(self, query: str) -> str:
        """Search the KB and return formatted results."""
        # Use similarity_search_with_score instead
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=6)
        
        if not docs_and_scores:
            return "No relevant KB entries found. Confidence: 0.0"
        
        # Convert scores to relevance (higher is better, invert if needed)
        # FAISS returns distance scores (lower = more similar)
        # Convert to similarity score (0-1 range)
        results = []
        for doc, score in docs_and_scores:
            # Normalize score to 0-1 range (assuming max distance ~2.0)
            relevance = max(0, 1 - (score / 2.0))
            results.append((doc, relevance))
        
        context = "\n\n".join(
            f"[Confidence: {relevance:.3f}] {doc.page_content}"
            for doc, relevance in results
        )
        
        highest_confidence = max(relevance for _, relevance in results)
        
        result = f"=== KB Search Results ===\nHighest Match Confidence: {highest_confidence:.3f}\n\n{context}"
        return result


# Create instance
search_kb = SearchKBTool()