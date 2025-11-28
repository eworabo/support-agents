from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped
from typing import Optional

class Base(AsyncAttrs, DeclarativeBase):
    pass

class Ticket(Base):
    __tablename__ = 'tickets'
    id: Mapped[int] = Column(Integer, primary_key=True)
    status: Mapped[str] = Column(String(50), nullable=False)  # 'resolved', 'escalated', 'pending'
    content: Mapped[Optional[str]] = Column(Text)  # Ticket text
    summary: Mapped[Optional[str]] = Column(String(255))  # Escalation summary
    priority: Mapped[Optional[str]] = Column(String(50))  # 'Low', 'Medium', 'High', 'Urgent'
    department: Mapped[Optional[str]] = Column(String(50))  # 'Tier 2 Support'
    tag: Mapped[Optional[str]] = Column(String(50))  # 'bug', 'refund', etc.

class KBEntry(Base):
    __tablename__ = 'kb_entries'
    id: Mapped[int] = Column(Integer, primary_key=True)
    title: Mapped[str] = Column(String(255), nullable=False)
    content: Mapped[str] = Column(Text, nullable=False)
    file_url: Mapped[Optional[str]] = Column(String(255))  # URL to uploaded attachment