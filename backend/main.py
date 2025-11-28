from typing import List  
from aiohttp import ClientSession
import os
import sys

# Path for subprocess – adds root and backend explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))  # Backend folder itself

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
# Force dotenv from project root – fixes subprocess working dir quirk
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))


from openai import OpenAI
from fastapi import FastAPI, HTTPException, Depends, Query, File, Form, UploadFile
from models import Ticket, KBEntry
from pydantic import BaseModel, Field
from typing import Optional, Literal
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from tools import search_kb
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
import logging
from fastapi.staticfiles import StaticFiles
 # Import from your new models.py in backend/
# Imports now work – classifier.py is in backend folder, so relative path is fine
from classifier import classify_ticket
# from tools import search_kb  # Temporarily disabled while fixing tool format

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Async DB setup for scalable queries
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in .env")
engine = create_async_engine(DATABASE_URL, echo=True)  # echo=False in prod
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)  # Only log errors in production
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


# ================== MODELS ==================

class TicketInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=2000, description="Customer support ticket text")

class ClassificationResponse(BaseModel):
    tag: str = Field(..., description="Predicted category (bug, refund, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")

class ResolveOutput(BaseModel):
    resolved: bool
    reply: Optional[str] = None
    confidence: float

class EscalateOutput(BaseModel):
    summary: str
    suggested_department: str = "Tier 2 Support"
    priority: Literal["Low", "Medium", "High", "Urgent"] = "Medium"

class ResolveResponse(BaseModel):
    status: Literal["resolved", "escalated"]
    reply: Optional[str] = None
    summary: Optional[str] = None
    confidence: Optional[float] = None
    suggested_department: Optional[str] = None
    priority: Optional[str] = None
    tag: str

class TicketOut(BaseModel):
    id: int
    status: str
    summary: str | None
    priority: str | None
    department: str | None
    tag: str | None

class TicketDetail(TicketOut):
    content: str | None

class KBEntryOut(BaseModel):
    id: int
    title: str
    content: str

class KBEntryIn(BaseModel):
    title: str
    content: str

# ================== AGENTS & TASKS ==================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

resolver_agent = Agent(
    role="Senior Ticket Resolver",
    goal="Resolve only when KB match ≥0.82. Never guess or hallucinate.",
    backstory="10+ years in support, knows every edge case of our product.",
    tools=[search_kb],
    llm=llm,
    allow_delegation=False,
    verbose=True
)

escalator_agent = Agent(
    role="Escalation Specialist",
    goal="Write perfect handoff summaries that humans solve in <5 min.",
    backstory="Former Zendesk lead, writes summaries that managers love.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

resolve_task = Task(
    description="""
    1. Use search_kb tool (multiple calls allowed).
    2. Only if highest_confidence ≥ 0.82 craft polite, complete reply.
    3. Otherwise resolved=false.
    Ticket: {ticket_text}
    Tag: {tag} (class confidence {class_conf:.2f})
    """,
    agent=resolver_agent,
    expected_output="JSON matching ResolveOutput",
    output_pydantic=ResolveOutput
)

escalate_task = Task(
    description="""
    Create concise, actionable 2-3 sentence summary with issue, tag, key details, sentiment.
    Ticket: {ticket_text}
    Tag: {tag}
    """,
    agent=escalator_agent,
    expected_output="JSON matching EscalateOutput",
    output_pydantic=EscalateOutput
)

# ================================== FASTAPI APP ==================================

app = FastAPI(title="Support Swarm Backend")
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:3000", "*"],  
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# ADD THESE 3 LINES HERE (after line 163)
os.makedirs("uploads", exist_ok=True)  
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.post("/classify", response_model=ClassificationResponse)
async def classify_endpoint(ticket: TicketInput):
    try:
        result = classify_ticket(ticket.text)
        return ClassificationResponse(tag=result["tag"], confidence=result["confidence"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/resolve", response_model=ResolveResponse)
async def resolve_endpoint(ticket: TicketInput):
    try:
        classification = classify_ticket(ticket.text)
        
        crew = Crew(agents=[resolver_agent], tasks=[resolve_task], process=Process.sequential, verbose=True)
        
        crew_output = crew.kickoff(inputs={
            "ticket_text": ticket.text,
            "tag": classification["tag"],
            "class_conf": classification["confidence"]
        })

        # Access the pydantic output
        result = crew_output.pydantic

        if result.resolved:
            return ResolveResponse(
                status="resolved",
                reply=result.reply,
                confidence=result.confidence,
                tag=classification["tag"]
            )
        
        # Auto-escalate
# Auto-escalate
        esc_crew = Crew(agents=[escalator_agent], tasks=[escalate_task], process=Process.sequential, verbose=True)
        esc_output = esc_crew.kickoff(inputs={
            "ticket_text": ticket.text,
            "tag": classification["tag"]
        })
        esc_result = esc_output.pydantic  # ← Add this line

        return ResolveResponse(
            status="escalated",
            summary=esc_result.summary,
            suggested_department=esc_result.suggested_department,
            priority=esc_result.priority,
            tag=classification["tag"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resolution failed: {str(e)}")

@app.post("/escalate")
async def escalate_endpoint(ticket: TicketInput):
    try:
        classification = classify_ticket(ticket.text)
        crew = Crew(agents=[escalator_agent], tasks=[escalate_task], process=Process.sequential)

        result = crew.kickoff(inputs={"ticket_text": ticket.text, "tag": classification["tag"]})
        return {
            "status": "escalated",
            "summary": result.summary,
            "suggested_department": result.suggested_department,
            "priority": result.priority,
            "tag": classification["tag"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Escalation failed: {str(e)}")

@app.get("/metrics")
async def get_metrics(db: AsyncSession = Depends(get_db)):
    try:
        resolved = (await db.execute(select(func.count(Ticket.id)).where(Ticket.status.in_(['auto_resolved', 'human_resolved'])))).scalar() or 0
        escalated = (await db.execute(select(func.count(Ticket.id)).where(Ticket.status == 'escalated'))).scalar() or 0
        total_tickets = resolved + escalated
        resolve_rate = round((resolved / total_tickets * 100), 1) if total_tickets > 0 else 0.0
        escalate_rate = round((escalated / total_tickets * 100), 1) if total_tickets > 0 else 0.0
# Rest unchanged

        return {
            'resolveRate': resolve_rate,
            'escalateRate': escalate_rate,
            'totalTickets': total_tickets,
            'resolved': resolved,
            'escalated': escalated
        }
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch metrics")

@app.get("/escalate")
async def get_escalate_list(page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100), db: AsyncSession = Depends(get_db)):
    try:
        stmt = select(Ticket).where(Ticket.status == 'escalated').offset((page - 1) * size).limit(size)
        results = await db.execute(stmt)
        tickets = results.scalars().all()
        total_stmt = select(func.count(Ticket.id)).where(Ticket.status == 'escalated')
        total = (await db.execute(total_stmt)).scalar() or 0
        return {"items": tickets, "total": total, "page": page, "size": size, "pages": (total + size - 1) // size}
    except Exception as e:
        logger.error(f"Escalate list error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch escalated tickets")

@app.get("/escalate/{id}", response_model=TicketDetail)
async def get_escalate_detail(id: int, db: AsyncSession = Depends(get_db)):
    try:
        stmt = select(Ticket).where(Ticket.id == id, Ticket.status == 'escalated')
        result = await db.execute(stmt)
        ticket = result.scalar_one_or_none()
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return ticket
    except Exception as e:
        logger.error(f"Escalate detail error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch ticket detail")

@app.get("/kb")
async def get_kb_list(page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100), db: AsyncSession = Depends(get_db)):
    try:
        stmt = select(KBEntry).offset((page - 1) * size).limit(size)
        results = await db.execute(stmt)
        entries = results.scalars().all()
        total_stmt = select(func.count(KBEntry.id))
        total = (await db.execute(total_stmt)).scalar() or 0
        return {"items": entries, "total": total, "page": page, "size": size, "pages": (total + size - 1) // size}
    except Exception as e:
        logger.error(f"KB list error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch KB entries")


@app.post("/kb")
async def save_kb_entry(
    title: str = Form(...), 
    content: str = Form(...), 
    files: List[UploadFile] = File(None),  # Changed from file to files (List)
    db: AsyncSession = Depends(get_db)
):
    try:
        file_urls = []
        if files:
            os.makedirs("uploads", exist_ok=True)
            for file in files:
                file_path = f"uploads/{file.filename}"
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                file_urls.append(f"http://localhost:8000/{file_path}")

        file_url_str = ",".join(file_urls) if file_urls else None
        
        new_entry = KBEntry(title=title, content=content, file_url=file_url_str)
        db.add(new_entry)
        await db.commit()
        await db.refresh(new_entry)
        
        return new_entry
    except Exception as e:
        await db.rollback()
        logger.error(f"KB save error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save KB entry: {str(e)}")

@app.post("/tickets/{id}/resolve")
async def resolve_human_ticket(id: int, db: AsyncSession = Depends(get_db)):
    try:
        stmt = select(Ticket).where(Ticket.id == id, Ticket.status == 'escalated')
        result = await db.execute(stmt)
        ticket = result.scalar_one_or_none()
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found or not escalated")
        ticket.status = 'human_resolved'
        await db.commit()
        return {"status": "human_resolved"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to resolve ticket: {str(e)}")

