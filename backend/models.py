from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase

class Base(AsyncAttrs, DeclarativeBase):
    pass

class Ticket(Base):
    __tablename__ = 'tickets'
    id = Column(Integer, primary_key=True)
    status = Column(String(50), nullable=False)  # e.g., 'resolved', 'escalated', 'pending'
    # Add more fields here as your project grows, e.g.:
    # user_id = Column(Integer)
    # created_at = Column(DateTime, default=func.now())
    # content = Column(Text)