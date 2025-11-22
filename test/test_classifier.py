import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from backend.classifier import classify_ticket

dummy_tickets = [
    ("App crashes when I tap the profile tab", "bug"),
    ("Can you add export to PDF please?", "feature_request"),
    ("I want a full refund, this doesn't work", "refund"),
    ("Charged twice for December", "billing_issue"),
    ("Forgot password and reset link not arriving", "account_issue"),
    ("How do I invite team members?", "general_inquiry"),
    ("The search function returns no results ever", "bug"),
    ("It would be great to have dark mode", "feature_request"),
    ("Cancel my subscription and refund", "refund"),
    ("Payment failed but money was taken", "billing_issue"),
    ("Can't verify my email address", "account_issue"),
    ("When is the next update coming?", "general_inquiry"),
    ("Button does nothing when clicked", "bug"),
    ("Please add integration with Zapier", "feature_request"),
    ("I accidentally bought pro, refund please", "refund"),
    ("Invoice shows wrong amount", "billing_issue"),
    ("2FA code not being accepted", "account_issue"),
    ("How does the new AI feature work?", "general_inquiry"),
    ("The app freezes on iOS 18", "bug"),
    ("Need bulk user import via CSV", "feature_request"),
]

correct = 0
times = []

print("Running 20 real-world ticket classification tests...\n")

for text, expected in dummy_tickets:
    start = time.time()
    try:
        result = classify_ticket(text)
        elapsed = time.time() - start
        times.append(elapsed)
        if result == expected:
            correct += 1
            print(f"✓ {result:18} {elapsed:5.2f}s  | {text}")
        else:
            print(f"✗ GOT: {result:14} EXPECTED: {expected}  | {text}")
    except Exception as e:
        print(f"ERROR: {e}  | {text}")

print("\n" + "="*60)
print(f"ACCURACY: {correct}/{len(dummy_tickets)} ({correct/len(dummy_tickets)*100:.1f}%)")
print(f"AVG TIME: {sum(times)/len(times):.3f}s   MAX: {max(times):.3f}s")
print(f"MODEL: gpt-4o-mini @ temperature=0.0")
print("="*60)