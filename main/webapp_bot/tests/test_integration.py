import io, json, pytest, itertools

@pytest.mark.parametrize("route", ["/voice/tts", "/voice/embed"])
def test_requires_payload(client, route):
    r = client.post(route, json={})
    assert r.status_code == 400

def test_strike_blocks_tts(client, fake_user):
    # вручную вызываем внутренний метод add_strike 6 раз
    from server_bot import add_strike, MAX_STRIKES
    for _ in range(MAX_STRIKES):
        add_strike(fake_user)
    data = {"userId": fake_user, "text": "test", "slot": 0}
    r = client.post("/voice/tts", json=data)
    assert r.status_code == 403 or r.get_json()["status"] == "error"
