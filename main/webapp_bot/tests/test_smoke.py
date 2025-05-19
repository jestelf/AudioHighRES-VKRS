import json

def test_index_ok(client):
    r = client.get("/")
    assert r.status_code == 200 and b"<title" in r.data

def test_audio_check_missing(client):
    r = client.post("/audio_check")
    body = r.get_json()
    assert r.status_code == 400 and body["status"] == "error"
