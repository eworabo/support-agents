import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables first
load_dotenv()

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment. Check your .env file.")

kb_entries = [
    "Refund policy: Full refund within 30 days if product unused and order ID provided. Automatic for digital goods within 7 days.",
    "Login failure - clear browser cache, disable VPN, try incognito mode. If persists, reset password via email link.",
    "Payment declined - common causes: insufficient funds, expired card, CVV mismatch, or bank block. Advise trying PayPal or different card.",
    "Shipping delay >5 days - provide tracking link and offer $10 goodwill credit. Escalate if >10 days.",
    "Bug: App crashes on iOS 18 when opening profile - fixed in v2.4.1, instruct user to update app.",
    "Billing: Charged twice - locate duplicate transaction IDs and issue immediate refund for the extra charge.",
    "Account locked after 5 failed logins - automatic unlock after 30 min or send password reset.",
    "Feature request - politely acknowledge, say 'we're always improving' and log internally. Never promise timeline.",
    "Cancellation request - subscriptions can be cancelled anytime with pro-rated refund if within billing cycle.",
    "Bug: Checkout button grayed out on mobile - caused by ad blocker, disable or use desktop.",
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.from_texts(kb_entries, embeddings)