import requests
import json
import os
import random
import time

# CONFIG
API_URL = "http://localhost:8000/resolve"
API_KEY = os.getenv("API_KEY", "yourservicekey") # Make sure this matches your .env

headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY
}

# SCENARIOS
# 1. High Confidence (Should Auto-Resolve if KB exists)
# 2. Low Confidence (Should Escalate)
scenarios = [
    # BUG (High Clarity)
    "I am getting a 500 error when I try to upload a PNG file to the profile settings. It works fine with JPG.",
    # REFUND (Clear Policy)
    "I bought the subscription yesterday but I realized I don't need it. Can I get a refund? It's been less than 24 hours.",
    # ESCALATION (Complex/Vague)
    "My enterprise dashboard is showing completely wrong data compared to the export. I need to speak to a manager immediately, this is costing us money.",
    # FEATURE REQUEST
    "It would be really nice if we could have a dark mode for the mobile app.",
    # BILLING (Confusing)
    "I see two charges on my card but I only have one account. One is for $10 and one is for $50. help?"
]

print(f"🚀 Starting Ticket Population to {API_URL}...")

for i, ticket_text in enumerate(scenarios):
    payload = {"text": ticket_text}
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Ticket {i+1} Queued: {data['task_id']} (Tag: {data.get('tag')})")
        else:
            print(f"❌ Error sending ticket {i+1}: {response.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

    # Wait a bit to verify 'Live Metrics' updates in dashboard
    time.sleep(2) 

print("\n🎉 Population complete. Check your Celery terminal for processing logs.")