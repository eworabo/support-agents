"""
Migration Script: Fix Tag Names in Database
Run this script once to normalize all tags in your database.

Usage:
    python migrate_tags.py
"""

import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from models import Ticket

load_dotenv()

async def migrate_tags():
    """Normalize all tags in the database"""
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not found in .env file")
    
    engine = create_async_engine(DATABASE_URL, echo=True)
    AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    
    # Tag mappings: old -> new
    tag_mappings = {
        "general_inquiry": "general",
        "billing_issue": "billing",
        "feature_request": "feature",
        "account_issue": "account"
    }
    
    async with AsyncSessionLocal() as session:
        print("🔄 Starting tag migration...")
        
        # Get current tag distribution
        result = await session.execute(
            select(Ticket.tag, Ticket.id).order_by(Ticket.id)
        )
        all_tickets = result.all()
        
        print(f"\n📊 Found {len(all_tickets)} tickets")
        
        # Count tags before migration
        from collections import Counter
        before_counts = Counter(tag for tag, _ in all_tickets)
        print("\n📊 Tag distribution BEFORE migration:")
        for tag, count in before_counts.items():
            print(f"   {tag}: {count}")
        
        # Update each old tag to new format
        updated_count = 0
        for old_tag, new_tag in tag_mappings.items():
            stmt = update(Ticket).where(Ticket.tag == old_tag).values(tag=new_tag)
            result = await session.execute(stmt)
            count = result.rowcount
            if count > 0:
                print(f"✅ Updated {count} tickets: {old_tag} → {new_tag}")
                updated_count += count
        
        await session.commit()
        
        # Get tag distribution after migration
        result = await session.execute(
            select(Ticket.tag, Ticket.id)
        )
        all_tickets_after = result.all()
        after_counts = Counter(tag for tag, _ in all_tickets_after)
        
        print(f"\n✅ Migration complete! Updated {updated_count} tickets")
        print("\n📊 Tag distribution AFTER migration:")
        for tag, count in after_counts.items():
            print(f"   {tag}: {count}")
        
        print("\n🎉 All tags are now standardized!")

if __name__ == "__main__":
    asyncio.run(migrate_tags())
