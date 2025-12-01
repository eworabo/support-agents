# In a new file or extend migrate_to_supabase.py
from langchain_openai import OpenAIEmbeddings
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings()  # Needs your OPENAI_API_KEY in .env

# Fetch all KB entries without embeddings
res = supabase.table('kb_entries').select('*').is_('embedding', 'null').execute()
for entry in res.data:
    text = f"{entry['title']}: {entry['content']}"
    emb = embeddings.embed_query(text)
    supabase.table('kb_entries').update({'embedding': emb}).eq('id', entry['id']).execute()

print("Embeddings added! Vectors ready to rumble.")