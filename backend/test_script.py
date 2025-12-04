from locust import HttpUser, task, between

class SupportSwarmUser(HttpUser):
    wait_times = between(1, 3)  # Simulate real user waits

    @task
    def resolve_ticket(self):
        self.client.post("/resolve", json={"text": "Dummy ticket about a bug"}, headers={"api-key": "yoursecretkey"})  # Your key from .env