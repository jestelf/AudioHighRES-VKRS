from locust import HttpUser, task, between
class LoadTest(HttpUser):
    wait_time = between(0.05, 0.2)   # 5-20 rps per user

    @task(3)
    def index(self): self.client.get("/")

    @task(1)
    def tts(self):
        self.client.post("/voice/tts", json={
            "userId": "locust",
            "text": "Привет!",
            "slot": 0
        })
