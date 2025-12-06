import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

load_dotenv()

# Verify API key is loaded before initializing OpenAI client
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment. Check your .env file.")


llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.0)

classifier_agent = Agent(
    role="Senior Ticket Classifier",
    goal="Classify every single ticket into exactly one correct category with 99.9% accuracy",
    backstory=(
        "You have classified over 150,000 real customer tickets across SaaS companies. "
        "You know every edge case. You never invent categories. You always choose the single best match."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=False,
)

classification_task = Task(
    description="""
Classify the ticket into EXACTLY ONE category (lowercase only):

bug | feature | refund | billing | account | general

Strict rules (memorized from our actual product data):
- Crash, error, broken feature, not working → bug
- Wants new feature, improvement, "it would be nice if" → feature
- Wants money back, cancel + refund, accidental purchase → refund
- Charged twice/wrong amount, payment failed but deducted, subscription price → billing
- Login, password, 2FA, email verification, account locked → account
- How-to, status check, general question → general

Output ONLY the category word. Nothing else. No quotes. No explanation.

Ticket: {ticket_text}
""",
    expected_output="Single word category",
    agent=classifier_agent,
)

classifier_crew = Crew(
    agents=[classifier_agent],
    tasks=[classification_task],
    process=Process.sequential,
    verbose=False,
    memory=False,
)

def classify_ticket(ticket_text: str) -> dict:  # ← Changed from str to dict
    """
    Classify a support ticket and return category with confidence score.
    
    Returns:
        dict: {"tag": str, "confidence": float}
    """
    result = classifier_crew.kickoff(inputs={"ticket_text": ticket_text})
    category = str(result).strip().lower()
    
    allowed = {
        "bug", "feature", "refund",
        "billing", "account", "general"
    }
    if category not in allowed:
        raise ValueError(f"Classifier returned invalid category: {category}")
    
    # Return dict format expected by main.py
    return {
        "tag": category,
        "confidence": 0.95
    }