import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine, text
from supabase import create_client
from langchain_openai import OpenAIEmbeddings

# Your Supabase creds from .env or dashboard
SUPABASE_URL = os.getenv('SUPABASE_URL')  # e.g., https://yourproject.supabase.co
SUPABASE_KEY = os.getenv('SUPABASE_KEY')  # Your service key

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings()

# Your old Postgres URL – boom, plugged in!
OLD_DB_URL = 'postgresql://postgres:postgres@localhost/support_agents_db'  # Switched to sync for migration stability; add +psycopg2 if SQLAlchemy gripes

old_engine = create_engine(OLD_DB_URL)

# Pull from old KB entries table (adjust if your table/columns differ)
with old_engine.connect() as conn:
    # Fetch all – for big DBs, batch it with LIMIT/OFFSET loops to avoid memory bombs
    old_entries = conn.execute(text("SELECT * FROM kb_entries")).fetchall()  # Tweak SELECT if needed

# Insert to Supabase
for entry in old_entries:
    data = {
        'title': entry.title,  # Map your actual column names here
        'content': entry.content,
        'file_url': entry.file_url if hasattr(entry, 'file_url') else None,  # Handle if missing
        # Add more fields as per your schema
    }
    supabase.table('kb_entries').insert(data).execute()

# Repeat for tickets table if needed
with old_engine.connect() as conn:
    old_tickets = conn.execute(text("SELECT * FROM tickets")).fetchall()

for ticket in old_tickets:
    data = {
        'status': ticket.status,
        'content': ticket.content,
        'summary': ticket.summary,
        'priority': ticket.priority,
        'department': ticket.department,
        'tag': ticket.tag,
    }
    supabase.table('tickets').insert(data).execute()

print("Migration complete! Check Supabase dashboard for your data.")