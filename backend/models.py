from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase

class Base(AsyncAttrs, DeclarativeBase):
    pass

class Ticket(Base):
    __tablename__ = 'tickets'
    id = Column(Integer, primary_key=True)
    status = Column(String(50), nullable=False)  # 'resolved', 'escalated', 'pending'
    content = Column(Text)  # Customer ticket text
    summary = Column(String(255))  # Escalation summary
    priority = Column(String(50))  # 'Low', 'Medium', 'High', 'Urgent'
    department = Column(String(50))  # 'Tier 2 Support'
    tag = Column(String(50))  # 'bug', 'refund', etc.