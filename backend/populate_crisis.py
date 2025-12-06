import requests
import random
import time
from dotenv import load_dotenv
import os

load_dotenv() 
# URL of your local API
API_URL = "http://localhost:8000/resolve"
API_KEY = os.getenv("API_KEY")# M

# These are vague, angry, or complex queries the AI *shouldn't* answer comfortably
hard_tickets = [
    "Your application deleted my entire database! I'm losing millions!",
    "I found a critical security vulnerability in your API, contact me immediately.",
    "The server is down and I'm getting a 504 Gateway Time-out.",
    "I want to speak to a manager right now. This is unacceptable.",
    "My custom integration with SAP is failing on line 403 of the legacy codebase."
    "I'm having trouble with the new AI feature, it's not working as expected.",
]



print("🔥 Starting Crisis Simulation...")

for i, ticket in enumerate(hard_tickets):
    payload = {"text": ticket}
    headers = {"api-key": API_KEY}
    
    requests.post(API_URL, json=payload, headers=headers)
    print(f"🚨 [Ticket {i+1}] Escalation candidate sent.")
    time.sleep(0.5)

print("🔥 Simulation Complete! Check the Escalation Queue.")