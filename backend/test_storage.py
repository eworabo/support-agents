"""
Test script to verify Supabase storage is working.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def test_storage():
    print("🧪 Testing Supabase Storage\n")
    print("="*60)
    
    # Test 1: List buckets
    print("\n1️⃣ Testing bucket access...")
    try:
        buckets = supabase.storage.list_buckets()
        print(f"   ✅ Found {len(buckets)} storage buckets:")
        for bucket in buckets:
            print(f"      - {bucket['name']} (public: {bucket.get('public', False)})")
        
        # Check if kb_uploads exists
        kb_bucket_exists = any(b['name'] == 'kb_uploads' for b in buckets)
        if not kb_bucket_exists:
            print("\n   ⚠️ 'kb_uploads' bucket NOT FOUND!")
            print("   📝 Action needed:")
            print("      1. Go to Supabase Dashboard → Storage")
            print("      2. Create a new bucket called 'kb_uploads'")
            print("      3. Make it public OR configure access policies")
            return False
        else:
            print("\n   ✅ 'kb_uploads' bucket exists!")
            
    except Exception as e:
        print(f"   ❌ Error accessing storage: {e}")
        return False
    
    # Test 2: Try to upload a test file
    print("\n2️⃣ Testing file upload...")
    try:
        test_content = b"Test file content"
        test_filename = "test-upload.txt"
        
        upload_result = supabase.storage.from_('kb_uploads').upload(
            test_filename,
            test_content,
            {'content-type': 'text/plain'}
        )
        
        print(f"   ✅ Upload successful!")
        print(f"   File: {test_filename}")
        
        # Test 3: Get public URL
        print("\n3️⃣ Testing public URL generation...")
        try:
            url = supabase.storage.from_('kb_uploads').get_public_url(test_filename)
            print(f"   ✅ Public URL generated:")
            print(f"   {url}")
        except Exception as e:
            print(f"   ⚠️ Could not get public URL: {e}")
        
        # Test 4: Clean up
        print("\n4️⃣ Cleaning up test file...")
        try:
            supabase.storage.from_('kb_uploads').remove([test_filename])
            print("   ✅ Test file deleted")
        except Exception as e:
            print(f"   ⚠️ Could not delete test file: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        print("\n   📝 Common causes:")
        print("      1. Bucket doesn't exist")
        print("      2. No write permissions")
        print("      3. Bucket is not public and no RLS policies set")
        print("\n   💡 Fix:")
        print("      Go to Supabase Dashboard → Storage → kb_uploads")
        print("      → Configuration → Make bucket public")
        return False
    
    print("\n" + "="*60)

if __name__ == "__main__":
    success = test_storage()
    
    if success:
        print("\n🎉 Storage is working! You can now upload files.")
    else:
        print("\n❌ Storage needs configuration. Follow the steps above.")
