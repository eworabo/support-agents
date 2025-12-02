"""
Rebuild embeddings for all KB entries.
Run this after fixing your Supabase configuration.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment. Check your .env file.")


def rebuild_all_embeddings():
    """Rebuild embeddings for all KB entries."""
    
    print("🔄 Starting embedding rebuild process...\n")
    
    try:
        # Get all KB entries
        result = supabase.table('kb_entries').select('id, title, content').execute()
        entries = result.data
        
        if not entries:
            print("❌ No KB entries found. Add some entries first!")
            return
        
        print(f"📊 Found {len(entries)} KB entries")
        print("="*60)
        
        embeddings = OpenAIEmbeddings()
        success_count = 0
        fail_count = 0
        
        for i, entry in enumerate(entries, 1):
            entry_id = entry['id']
            title = entry['title']
            content = entry['content']
            
            print(f"\n[{i}/{len(entries)}] Processing entry {entry_id}: {title[:50]}...")
            
            try:
                # Generate embedding
                text = f"{title}: {content}"
                emb = embeddings.embed_query(text)
                
                # Update in Supabase
                update_result = supabase.table('kb_entries').update({
                    'embedding': emb
                }).eq('id', entry_id).execute()
                
                if update_result.data:
                    print(f"          ✅ Success (dimension: {len(emb)})")
                    success_count += 1
                else:
                    print(f"          ⚠️ Update returned no data")
                    fail_count += 1
                    
            except Exception as e:
                print(f"          ❌ Failed: {str(e)}")
                fail_count += 1
        
        print("\n" + "="*60)
        print("📊 REBUILD SUMMARY")
        print("="*60)
        print(f"✅ Successfully embedded: {success_count}/{len(entries)}")
        print(f"❌ Failed: {fail_count}/{len(entries)}")
        
        if fail_count > 0:
            print("\n⚠️ Some embeddings failed. Common causes:")
            print("  1. Embedding column not configured properly")
            print("  2. Run check_supabase_setup.py to fix database setup")
            print("  3. Check SUPABASE_SERVICE_KEY has write permissions")
        else:
            print("\n🎉 All embeddings generated successfully!")
            print("💡 Your search_kb tool should now work properly!")
        
        return success_count
        
    except Exception as e:
        print(f"\n❌ Fatal error during rebuild: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("  1. Check your .env file has SUPABASE_URL and SUPABASE_SERVICE_KEY")
        print("  2. Verify OPENAI_API_KEY is set")
        print("  3. Run check_supabase_setup.py to verify database configuration")
        return 0


if __name__ == "__main__":
    print("="*60)
    print("🚀 KB EMBEDDING REBUILD TOOL")
    print("="*60)
    print()
    
    # Confirm before proceeding
    print("This will generate embeddings for all KB entries.")
    print("⚠️  This uses OpenAI API credits (approximately $0.0001 per entry)")
    print()
    
    response = input("Continue? (yes/no): ").lower().strip()
    
    if response == 'yes' or response == 'y':
        print()
        rebuild_all_embeddings()
    else:
        print("❌ Cancelled by user")
