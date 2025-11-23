from crewai.tools import Tool
from kb_vectorstore import vectorstore

def _search_kb_func(query: str) -> dict:
    """
    Search the knowledge base for relevant support information.
    Returns KB entries with relevance scores (0-1 cosine similarity).
    """
    docs_and_scores = vectorstore.similarity_search_with_relevance_score(query, k=6)
    if not docs_and_scores:
        return {"context": "", "highest_confidence": 0.0, "docs": []}
    
    context = "\n\n".join(
        f"[Score: {score:.3f}] {doc.page_content}"
        for doc, score in docs_and_scores
    )
    highest_confidence = max(score for _, score in docs_and_scores)
    
    return {
        "context": context,
        "highest_confidence": round(highest_confidence, 3),
        "docs": [doc.page_content for doc, _ in docs_and_scores]
    }

# Create CrewAI Tool instance
search_kb = Tool(
    name="Search Knowledge Base",
    description="Search the knowledge base for relevant support information. Returns KB entries with relevance scores (0-1 cosine similarity).",
    func=_search_kb_func
)