import requests
import random
import time
from dotenv import load_dotenv
import os

load_dotenv() 
# URL of your local API
API_URL = "http://localhost:8000/resolve"
API_KEY = os.getenv("API_KEY")# 

tickets = [
    # Bug
    "The export to PDF button is grayed out on Mac.",
    # Refund
    "I was charged twice for the pro plan.",
    # Feature Request
    "It would be great if you had a dark mode.",
    # General
    "Are you open on public holidays?",
    # Billing
    "My credit card expired, how do I update it?"
]

print("🌍 Starting Real-World Simulation (Looping)...")

# Run for 1 minute
end_time = time.time() + 60

while time.time() < end_time:
    ticket = random.choice(tickets)
    payload = {"text": ticket}
    headers = {"api-key": API_KEY}
    
    requests.post(API_URL, json=payload, headers=headers)
    print(f"📨 Sent random ticket: {ticket[:20]}...")
    
    # Random sleep between 2s and 5s
    time.sleep(random.uniform(2, 5))

print("🌍 Simulation finished.")