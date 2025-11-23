import os
import sys

# Path for subprocess – adds root and backend explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))  # Backend folder itself

from dotenv import load_dotenv

# Force dotenv from project root – fixes subprocess working dir quirk
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))

from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from tools import search_kb
# Imports now work – classifier.py is in backend folder, so relative path is fine
from classifier import classify_ticket
# from tools import search_kb  # Temporarily disabled while fixing tool format

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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