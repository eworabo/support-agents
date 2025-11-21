from fastapi import FastAPI
from dotenv import load_dotenv
import os
from openai import OpenAI
from typing import Dict

load_dotenv()  # Env first—pulls key

# Debug print (remove after good)
print("Loaded API Key:", os.getenv("OPENAI_API_KEY") or "NOT SET!")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Client next

app = FastAPI(title="Support Swarm Backend")  # App def here—before decorators!

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