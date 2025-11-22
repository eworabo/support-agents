from classifier import classify_ticket
from pydantic import BaseModel, Field

from fastapi import FastAPI
from dotenv import load_dotenv
import os
from openai import OpenAI
from typing import Dict

load_dotenv()  # Env first—pulls key

# Debug print (remove after good)
print("Loaded API Key:", os.getenv("OPENAI_API_KEY") or "NOT SET!")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Client next

class TicketInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=2000)

class ClassificationResponse(BaseModel):
    category: str
    confidence: str = "high"

app = FastAPI(title="Support Swarm Backend")  # App def here—before decorators!

@app.post("/classify", response_model=ClassificationResponse)
async def classify(ticket: TicketInput):
    try:
        category = classify_ticket(ticket.text)
        return ClassificationResponse(category=category)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "message": "Underdog rising—proving them wrong!"}

@app.get("/test-openai")
async def test_openai() -> Dict[str, str]:
    try:
        client.models.list()  # Connection check
        return {"status": "connected"}
    except Exception as e:
        return {"error": str(e)}