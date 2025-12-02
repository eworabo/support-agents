import os
import requests
from io import BytesIO
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from supabase import create_client

# File processing libraries
try:
    from pypdf import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("⚠️ pypdf not installed. PDF extraction disabled. Install with: pip install pypdf")

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("⚠️ python-docx not installed. Word doc extraction disabled. Install with: pip install python-docx")

try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("⚠️ OCR libraries not installed. Image text extraction disabled.")
    print("   Install with: pip install pillow pytesseract")
    print("   Also install Tesseract: brew install tesseract (Mac) or apt-get install tesseract-ocr (Linux)")

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment. Check your .env file.")


def extract_text_from_pdf(file_url):
    """Extract text from PDF file."""
    if not HAS_PDF:
        return ""
    
    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        
        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        
        extracted = "\n".join(text).strip()
        print(f"   📄 Extracted {len(extracted)} chars from PDF")
        return extracted
        
    except Exception as e:
        print(f"   ⚠️ PDF extraction error: {str(e)}")
        return ""


def extract_text_from_docx(file_url):
    """Extract text from Word document."""
    if not HAS_DOCX:
        return ""
    
    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        
        docx_file = BytesIO(response.content)
        doc = Document(docx_file)
        
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        
        extracted = "\n".join(text).strip()
        print(f"   📝 Extracted {len(extracted)} chars from Word doc")
        return extracted
        
    except Exception as e:
        print(f"   ⚠️ Word doc extraction error: {str(e)}")
        return ""


def extract_text_from_image(file_url):
    """Extract text from image using OCR."""
    if not HAS_OCR:
        return ""
    
    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        text = pytesseract.image_to_string(image)
        
        extracted = text.strip()
        print(f"   🖼️ Extracted {len(extracted)} chars from image (OCR)")
        return extracted
        
    except Exception as e:
        print(f"   ⚠️ Image OCR error: {str(e)}")
        return ""


def extract_text_from_file(file_url):
    """
    Extract text from attached file based on file extension.
    Supports: PDF, Word (.docx), Images (.jpg, .png, .jpeg, .gif, .webp)
    
    Args:
        file_url: URL to the file in Supabase storage
        
    Returns:
        Extracted text content, or empty string if extraction fails
    """
    if not file_url:
        return ""
    
    # Get file extension
    file_ext = file_url.lower().split('.')[-1].split('?')[0]
    
    print(f"   📎 Processing attachment: .{file_ext}")
    
    if file_ext == 'pdf':
        return extract_text_from_pdf(file_url)
    
    elif file_ext in ['docx', 'doc']:
        return extract_text_from_docx(file_url)
    
    elif file_ext in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'tiff']:
        return extract_text_from_image(file_url)
    
    else:
        print(f"   ⚠️ Unsupported file type: .{file_ext}")
        return ""


def embed_kb_entry(entry_id, title, content, file_url=None):
    """
    Generate and store embedding for a KB entry in Supabase.
    Now includes text extraction from attached files!
    
    Args:
        entry_id: The KB entry ID
        title: Entry title
        content: Entry content
        file_url: Optional URL to attached file (can be comma-separated for multiple files)
    """
    try:
        print(f"\n🔄 Embedding entry {entry_id}...")
        
        # Start with title and content
        text_parts = [f"{title}: {content}"]
        
        # Extract text from attached files if present
        if file_url:
            # Handle multiple files (comma-separated URLs)
            file_urls = [url.strip() for url in file_url.split(',')]
            
            for url in file_urls:
                extracted_text = extract_text_from_file(url)
                if extracted_text:
                    text_parts.append(extracted_text)
        
        # Combine all text
        full_text = "\n\n".join(text_parts)
        
        # Truncate if too long (OpenAI has token limits)
        MAX_CHARS = 30000  # Roughly 8k tokens
        if len(full_text) > MAX_CHARS:
            print(f"   ⚠️ Text too long ({len(full_text)} chars), truncating to {MAX_CHARS}")
            full_text = full_text[:MAX_CHARS]
        
        print(f"   📊 Total text length: {len(full_text)} chars")
        
        # Generate embedding
        embeddings = OpenAIEmbeddings()
        emb = embeddings.embed_query(full_text)
        
        print(f"   🧮 Generated embedding (dimension: {len(emb)})")
        
        # Update in Supabase
        result = supabase.table('kb_entries').update({
            'embedding': emb
        }).eq('id', entry_id).execute()
        
        if result.data:
            print(f"   ✅ Successfully embedded entry {entry_id}!")
        else:
            print(f"   ⚠️ Update returned no data for entry {entry_id}")
            
    except Exception as e:
        print(f"   ❌ Embed error for entry {entry_id}: {str(e)}")
        # Don't raise - we want KB entry to be saved even if embedding fails


def search_kb_entries(query_text, limit=5, threshold=0.7):
    """
    Search KB entries using vector similarity.
    
    Args:
        query_text: The search query
        limit: Number of results to return
        threshold: Minimum similarity score (0-1)
        
    Returns:
        List of matching KB entries with similarity scores
    """
    try:
        embeddings = OpenAIEmbeddings()
        query_emb = embeddings.embed_query(query_text)
        
        # Use Supabase RPC function for vector search
        result = supabase.rpc('match_kb_entries', {
            'query_embedding': query_emb,
            'match_threshold': threshold,
            'match_count': limit
        }).execute()
        
        return result.data
        
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        return []


def rebuild_all_embeddings():
    """
    Rebuild embeddings for all KB entries.
    Now includes file attachments!
    """
    try:
        # Get all KB entries including file_url
        result = supabase.table('kb_entries').select('id, title, content, file_url').execute()
        entries = result.data
        
        print(f"🔄 Rebuilding embeddings for {len(entries)} entries...\n")
        
        success_count = 0
        for entry in entries:
            embed_kb_entry(
                entry['id'], 
                entry['title'], 
                entry['content'],
                entry.get('file_url')  # Include file URL
            )
            success_count += 1
            
        print(f"\n✅ Successfully rebuilt {success_count}/{len(entries)} embeddings!")
        return success_count
        
    except Exception as e:
        print(f"❌ Rebuild error: {str(e)}")
        return 0