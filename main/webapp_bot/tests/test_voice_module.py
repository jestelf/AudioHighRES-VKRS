from server_bot import VOICE
import numpy as np, torch

def test_default_params():
    uid = "userX"
    p = VOICE.get_user_params(uid)
    assert 0.0 <= p["temperature"] <= 1.0

def test_clamp_speed():
    uid = "userX"
    VOICE.set_user_params(uid, speed=10)
    assert VOICE.get_user_params(uid)["speed"] <= 3.0

def test_embedding_roundtrip(tmp_path):
    dummy = tmp_path / "dummy.wav"
    dummy.write_bytes(b"RIFF....WAVEfmt ")
    uid = "77"
    path = VOICE.create_embedding(dummy, uid)
    assert path.exists()
    out = VOICE.synthesize(uid, "Привет", embedding_file=path)
    assert out.exists() and out.stat().st_size > 1000
