import requests
import random
import time
from dotenv import load_dotenv
import os

load_dotenv() 
# URL of your local API
API_URL = "http://localhost:8000/resolve"
API_KEY = os.getenv("API_KEY")# Match the API_KEY in your .env
print(f"🔑 API Key loaded: {API_KEY}")

# These queries should match content likely in your KB (e.g., "refund policy", "password reset")
easy_tickets = [
    "I need to reset my password, I forgot it.",
    "What is your refund policy for digital products?",
    "How do I update my billing address in the dashboard?",
    "Where can I find my invoice history?",
    "I want to cancel my subscription, how do I do that?",
    "I need help with the new AI feature, it's not working.",
    "I'm having trouble with the new AI feature, it's not working as expected.",
    "I'm having trouble with the new AI feature, it's not working as expected.",
]

print("🚀 Starting Happy Path Simulation...")

for i, ticket in enumerate(easy_tickets):
    payload = {"text": ticket}
    headers = {"api-key": API_KEY, "Content-Type": "application/json"}
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            print(f"✅ [Ticket {i+1}] Sent: '{ticket[:30]}...' -> Queued")
        else:
            print(f"❌ [Ticket {i+1}] Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"⚠️ Connection Error: {e}")
    
    # Wait a bit to simulate natural traffic flow
    time.sleep(1) 

print("✨ Simulation Complete! Check your dashboard.")