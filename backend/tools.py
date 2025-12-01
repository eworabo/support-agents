from crewai.tools import BaseTool
from typing import Type, Any
from pydantic import BaseModel, Field
import docx
from pymupdf import open as pdf_open
from pytesseract import image_to_string
from PIL import Image

class SearchKBInput(BaseModel):
    """Input schema for SearchKBTool."""
    query: str = Field(..., description="Search query for the knowledge base")

class SearchKBTool(BaseTool):
    name: str = "search_kb"
    description: str = "Searches the knowledge base for similar entries and returns matches with confidence, including extracted content from attached files like Word, PDF, or images."
    args_schema: Type[BaseModel] = SearchKBInput

    def _run(self, query: str) -> str:
        """Searches the knowledge base."""
        from main import vectorstore
        
        if vectorstore is None:
            return "Knowledge base not loaded."
        
        try:
            results = vectorstore.similarity_search_with_score(query, k=3)
            output = []
            
            for doc, score in results:
                confidence = 1 - score
                
                # Extract from file if present
                entry = doc.metadata.get('entry')
                if entry and hasattr(entry, 'file_url') and entry.file_url:
                    for url in entry.file_url.split(','):
                        url = url.strip()
                        try:
                            file_path = url.replace('http://localhost:8000/', '')
                            if url.endswith('.docx'):
                                doc_file = docx.Document(file_path)
                                extracted = '\n'.join(p.text for p in doc_file.paragraphs)
                            elif url.endswith('.pdf'):
                                with pdf_open(file_path) as pdf:
                                    extracted = ''.join(page.get_text() for page in pdf)
                            elif url.endswith(('.png', '.jpg', '.jpeg')):
                                img = Image.open(file_path)
                                extracted = image_to_string(img)
                            else:
                                extracted = ''
                            
                            if extracted:
                                doc.page_content += f"\nExtracted from file: {extracted}"
                        except Exception as e:
                            print(f"Error extracting from {url}: {e}")
                
                if confidence >= 0.82:
                    output.append(f"High-confidence match ({confidence:.2f}): {doc.page_content}")
                else:
                    output.append(f"Low-confidence match ({confidence:.2f}): {doc.page_content}")
            
            return "\n\n".join(output) if output else "No relevant matches found."
        except Exception as e:
            return f"Error searching KB: {str(e)}"

# Create instance
search_kb = SearchKBTool()