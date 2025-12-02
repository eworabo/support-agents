"""
Setup script to check and configure Supabase for vector embeddings.
Run this to ensure your database is properly configured.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')  # Need service key for SQL operations
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def check_and_setup_supabase():
    """Check if Supabase is properly configured for vector embeddings."""
    
    print("🔍 Checking Supabase configuration...")
    
    # Step 1: Check if pgvector extension is enabled
    print("\n1️⃣ Checking pgvector extension...")
    try:
        # This SQL enables the extension if it doesn't exist
        supabase.rpc('exec_sql', {'sql': 'CREATE EXTENSION IF NOT EXISTS vector;'}).execute()
        print("   ✅ pgvector extension enabled")
    except Exception as e:
        print(f"   ⚠️ Could not enable pgvector: {e}")
        print("   📝 Action needed: Go to Supabase SQL Editor and run:")
        print("      CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Step 2: Check if embedding column exists and is correct type
    print("\n2️⃣ Checking embedding column...")
    try:
        result = supabase.table('kb_entries').select('id, embedding').limit(1).execute()
        print("   ✅ embedding column exists")
    except Exception as e:
        print(f"   ⚠️ embedding column issue: {e}")
        print("   📝 Action needed: Go to Supabase SQL Editor and run:")
        print("      ALTER TABLE kb_entries ADD COLUMN embedding vector(1536);")
    
    # Step 3: Check if search function exists
    print("\n3️⃣ Checking search function...")
    try:
        # Try to call the function with dummy data (will fail if function doesn't exist)
        supabase.rpc('match_kb_entries', {
            'query_embedding': [0.0] * 1536,
            'match_threshold': 0.7,
            'match_count': 1
        }).execute()
        print("   ✅ match_kb_entries function exists")
    except Exception as e:
        print(f"   ⚠️ Search function not found: {e}")
        print("   📝 Action needed: Create the RPC function (see instructions below)")
    
    # Step 4: Check how many entries have embeddings
    print("\n4️⃣ Checking existing embeddings...")
    try:
        all_entries = supabase.table('kb_entries').select('id, embedding').execute()
        total = len(all_entries.data)
        with_embeddings = sum(1 for entry in all_entries.data if entry.get('embedding'))
        without_embeddings = total - with_embeddings
        
        print(f"   📊 Total entries: {total}")
        print(f"   ✅ With embeddings: {with_embeddings}")
        print(f"   ❌ Without embeddings: {without_embeddings}")
        
        if without_embeddings > 0:
            print(f"   💡 Run rebuild_embeddings.py to fix this")
            
    except Exception as e:
        print(f"   ⚠️ Could not check entries: {e}")
    
    print("\n" + "="*60)
    print("📋 SETUP INSTRUCTIONS")
    print("="*60)
    print("""
If any checks failed, follow these steps:

1. Go to your Supabase Dashboard → SQL Editor
2. Run the following SQL commands:

-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Ensure embedding column exists with correct type
ALTER TABLE kb_entries 
ADD COLUMN IF NOT EXISTS embedding vector(1536);

-- Create search function
CREATE OR REPLACE FUNCTION match_kb_entries(
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id bigint,
  title text,
  content text,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    kb_entries.id,
    kb_entries.title,
    kb_entries.content,
    1 - (kb_entries.embedding <=> query_embedding) as similarity
  FROM kb_entries
  WHERE kb_entries.embedding IS NOT NULL
    AND 1 - (kb_entries.embedding <=> query_embedding) > match_threshold
  ORDER BY kb_entries.embedding <=> query_embedding
  LIMIT match_count;
$$;

-- Create index for faster searches (optional but recommended)
CREATE INDEX IF NOT EXISTS kb_entries_embedding_idx 
ON kb_entries USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

3. After running the SQL, re-run this script to verify.

4. Run rebuild_embeddings.py to generate embeddings for existing entries.
    """)


if __name__ == "__main__":
    check_and_setup_supabase()
