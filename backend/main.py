from typing import List  
from aiohttp import ClientSession
import os
import sys

# Path for subprocess – adds root and backend explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))  # Backend folder itself

from celery import Celery 
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import re 

load_dotenv(dotenv_path=os.path.join(project_root, '.env'))
print("REDIS_URL check: ", os.getenv('REDIS_URL'))
app_celery = Celery('support_swarm', broker=os.getenv('REDIS_URL'), backend=os.getenv('REDIS_URL'))
print(app_celery)

from openai import OpenAI
from fastapi import FastAPI, HTTPException, Depends, Query, File, Form, UploadFile, Request, Header
from models import Ticket, KBEntry
from pydantic import BaseModel, Field
from typing import Optional, Literal
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from tools import search_kb
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
import logging
from fastapi.staticfiles import StaticFiles
from classifier import classify_ticket
from langchain_openai import OpenAIEmbeddings
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import resend
from supabase import create_client, Client
from kb_vectorstore import embed_kb_entry, rebuild_all_embeddings
import datetime


@app_celery.task
def run_resolve_task(ticket_text: str, tag: str, class_conf: float):
    # Rebuild everything fresh – no globals choking us
    from crewai import Agent, Task, Crew, Process
    from langchain_openai import ChatOpenAI
    from tools import search_kb  # Your KB tool
    from pydantic import BaseModel, Field
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

    crew = Crew(agents=[resolver_agent], tasks=[resolve_task], process=Process.sequential, verbose=True)
    crew_output = crew.kickoff(inputs={"ticket_text": ticket_text, "tag": tag, "class_conf": class_conf})

    #FIX: Force to plain, serializable dict - no Pydantic residue
    if hasattr(crew_output, 'pydantic') and crew_output.pydantic:
        result = {
            "resolved": crew_output.pydantic.resolved,
            "reply": crew_output.pydantic.reply,
            "confidence": crew_output.pydantic.confidence
        }
    else:
        result = {
            "resolved": False,
            "reply": None,
            "confidence": 0.0,
            "error": "Failed to generate structured output"
        }
    
    import json
    return json.loads(json.dumps(result))  # Nukes any non-JSON junk
    
    # Make sure we never return a ResolveOutput object, only a dict
    result_model = getattr(crew_output, "pydantic", None)
    
    if isinstance(result_model, BaseModel):
        # Pydantic v2 uses model_dump, v1 uses dict
        if hasattr(result_model, "model_dump"):
            return result_model.model_dump()
        else:
            return result_model.dict()
    else:
        # Fallback if the LLM failed to generate valid JSON
        return {
            "resolved": False, 
            "reply": None, 
            "confidence": 0.0, 
            "error": "Failed to generate structured output"
        } 


supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
# Quick test: print(supabase.table('kb_entries').select('count(*)').execute())  # Comment out after
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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

# NEW: Response model for async task queuing
class TaskQueueResponse(BaseModel):
    task_id: str
    message: str
    status: Literal["queued", "pending"]
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

# Startup event
@app.on_event("startup")
async def startup_event():
    print("🚀 FastAPI app started successfully")

@app.post("/classify", response_model=ClassificationResponse)
async def classify_endpoint(ticket: TicketInput):
    try:
        result = classify_ticket(ticket.text)
        return ClassificationResponse(tag=result["tag"], confidence=result["confidence"])
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/resolve", response_model=TaskQueueResponse)
async def resolve_endpoint(
    ticket: TicketInput,
    api_key: str = Header(None),
    db: AsyncSession = Depends(get_db)
):
    """Queue ticket for async resolution"""
    try:
        if api_key != os.getenv("API_KEY"):
            raise HTTPException(status_code=401, detail="Invalid API key")
            
        classification = classify_ticket(ticket.text)
        tag = classification["tag"]
        class_conf = classification["confidence"]
        
        new_ticket = Ticket(
            status='pending',
            content=ticket.text,
            tag=tag
        )
        db.add(new_ticket)
        await db.commit()
        await db.refresh(new_ticket)
        
        task = run_resolve_task.apply_async(
            args=[ticket.text, tag, class_conf],
            task_id=f"ticket_{new_ticket.id}"
        )
        
        return TaskQueueResponse(
            task_id=task.id,
            message="Ticket queued for processing",
            status="queued",
            tag=tag
        )
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Resolve endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue ticket: {str(e)}")

@app.get("/metrics")
async def get_metrics(db: AsyncSession = Depends(get_db)):
    """Return dashboard metrics"""
    try:
        result = await db.execute(select(
            func.count().filter(Ticket.status == 'resolved').label('resolved'),
            func.count().filter(Ticket.status == 'escalated').label('escalated'),
            func.count().label('total')
        ))
        row = result.one()
        
        total = row.total or 1
        resolved_count = row.resolved or 0
        escalated_count = row.escalated or 0
        
        return {
            "totalTickets": total,
            "resolved": resolved_count,
            "escalated": escalated_count,
            "resolveRate": round((resolved_count / total) * 100, 1) if total > 0 else 0,
            "escalateRate": round((escalated_count / total) * 100, 1) if total > 0 else 0,
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch metrics")

@app.get("/tickets/escalated")
async def list_escalated_tickets(db: AsyncSession = Depends(get_db)):
    """Return all escalated tickets"""
    try:
        result = await db.execute(
            select(Ticket).where(Ticket.status == 'escalated').order_by(Ticket.id.desc())
        )
        tickets = result.scalars().all()
        return {"items": [
            {
                "id": t.id,
                "summary": t.summary,
                "priority": t.priority,
                "department": t.department,
                "tag": t.tag
            } for t in tickets
        ]}
    except Exception as e:
        logger.error(f"Escalated tickets error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch escalated tickets")

@app.get("/tickets/{ticket_id}", response_model=TicketDetail)
async def get_ticket_detail(ticket_id: int, db: AsyncSession = Depends(get_db)):
    """Get full ticket details by ID"""
    try:
        result = await db.execute(select(Ticket).where(Ticket.id == ticket_id))
        ticket = result.scalar_one_or_none()
        
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        return TicketDetail(
            id=ticket.id,
            status=ticket.status,
            content=ticket.content,
            summary=ticket.summary,
            priority=ticket.priority,
            department=ticket.department,
            tag=ticket.tag
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ticket detail error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch ticket detail")

@app.get("/kb")
async def list_kb_entries(size: int = Query(50, ge=1, le=200)):
    """Return all KB entries for dashboard"""
    try:
        response = supabase.table("kb_entries")\
            .select("*")\
            .order("created_at", desc=True)\
            .limit(size)\
            .execute()

        items = response.data or []
        
        # Ensure file_url is always string (frontend expects it)
        for item in items:
            if item.get("file_url") is None:
                item["file_url"] = ""

        return {"items": items, "total": len(items)}
    except Exception as e:
        logger.error(f"KB list error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch KB entries")

@app.post("/kb")
async def create_kb_entry(
    title: str = Form(...), 
    content: str = Form(...), 
    files: List[UploadFile] = File(default=[])
):
    """Save new KB entry + upload files + generate embedding immediately"""
    try:
        file_urls = []
        
        # --- File Upload Logic ---
        for file in files:
            # Simple sanitization
            sanitized_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)
            file_content = await file.read()
            
            # Upload to Supabase storage
            upload_res = supabase.storage.from_('kb_uploads').upload(
                sanitized_filename, 
                file_content, 
                {'content-type': file.content_type}
            )
            # Get the public URL for the file
            file_url_data = supabase.storage.from_('kb_uploads').get_public_url(sanitized_filename)
            file_urls.append(file_url_data)
        
        file_urls_str = ",".join(file_urls)

        # --- DB Insertion Logic ---
        entry_data = {
            "title": title,
            "content": content,
            "file_url": file_urls_str,
            "embedding": None,  # Will be generated immediately
        }

        # Insert new entry
        res = supabase.table('kb_entries').insert(entry_data).execute()

        if res.data:
            entry_id = res.data[0]['id']
            
            # Generate embedding immediately (uses Supabase pgvector)
            embed_kb_entry(entry_id, title, content, file_urls_str)
            
            return {
                "status": "success", 
                "id": entry_id, 
                "message": "KB entry saved and embedded successfully"
            }
        else:
            raise Exception("Supabase insert failed to return data.")

    except Exception as e:
        logger.error(f"KB creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create KB entry: {str(e)}")

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

@app.post("/inbound_email")
async def inbound_email(request: Request, api_key: str = Header(None), db: AsyncSession = Depends(get_db)):
    """Handle inbound emails from SendGrid or similar"""
    try:
        if api_key != os.getenv("API_KEY"):
            raise HTTPException(status_code=401, detail="Invalid API key")
            
        payload = await request.json()  # SendGrid sends JSON with 'subject', 'text', 'from', etc.
        ticket_text = f"From: {payload.get('from')} Subject: {payload.get('subject')} Body: {payload.get('text')}"
        ticket = TicketInput(text=ticket_text)
        
        # Call resolve endpoint to queue the task
        response = await resolve_endpoint(ticket, api_key=api_key, db=db)
        
        return {"status": "processed", "task_id": response.task_id}
    except Exception as e:
        logger.error(f"Email process error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process email")


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Poll status of async Celery task"""
    from celery.result import AsyncResult
    
    try:
        task = AsyncResult(task_id, app=app_celery)
        
        if task.state == 'PENDING':
            return {"status": "pending", "message": "Task is queued or processing"}
        
        elif task.state == 'SUCCESS':
            result = task.result  # This is a DICT now
            
            # FIX: Check if it's a dict and access keys safely
            if isinstance(result, dict) and result.get('resolved'):
                return {
                    "status": "success",
                    "result": {
                        "status": "resolved",
                        "reply": result.get('reply'),
                        "confidence": result.get('confidence'),
                        "tag": "resolution"
                    }
                }
            else:
                # Task completed but ticket needs escalation
                confidence = result.get('confidence', 0.0) if isinstance(result, dict) else 0.0
                return {
                    "status": "needs_escalation",
                    "result": {
                        "confidence": confidence,
                        "message": "Ticket requires human escalation"
                    }
                }
        
        elif task.state == 'FAILURE':
            return {
                "status": "failed",
                "error": str(task.info)
            }
        
        else:
            return {
                "status": task.state.lower(),
                "info": str(task.info) if task.info else None
            }
            
    except Exception as e:
        logger.error(f"Task status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@app.delete("/kb/{entry_id}")
async def delete_kb_entry(entry_id: int):
    try:
        # 1. Retrieve the entry to get the file URL(s)
        entry_res = supabase.table('kb_entries').select('file_url').eq('id', entry_id).single().execute()
        entry = entry_res.data
        file_urls_str = entry.get('file_url')
        
        # 2. Delete the database entry
        supabase.table('kb_entries').delete().eq('id', entry_id).execute()
        
        # 3. Delete the file(s) from Supabase Storage
        if file_urls_str:
            file_urls = [url.strip() for url in file_urls_str.split(',') if url.strip()]
            
            # Extract filenames from URLs (e.g., from '.../kb_uploads/filename.docx' to 'filename.docx')
            filenames_to_delete = [url.split('/')[-1] for url in file_urls]
            
            if filenames_to_delete:
                # Supabase storage remove expects a list of paths
                supabase.storage.from_('kb_uploads').remove(filenames_to_delete)
        
        return {"status": "success", "message": f"KB entry {entry_id} and associated files deleted."}
        
    except Exception as e:
        # If the entry was not found or deletion failed
        if "No rows returned" in str(e):
             raise HTTPException(status_code=404, detail=f"KB entry {entry_id} not found.")
        logger.error(f"KB delete error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete KB entry: {str(e)}")