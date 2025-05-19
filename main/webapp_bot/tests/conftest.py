import pytest, os, tempfile, io, wave, contextlib
from server_bot import app as flask_app, VOICE

@pytest.fixture(scope="session")
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()

@pytest.fixture
def silence_wav(tmp_path):
    """1-секундный WAV 16 kHz silence."""
    path = tmp_path / "silence.wav"
    with contextlib.closing(wave.open(path, "w")) as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16_000)
        f.writeframes(b"\x00\x00" * 16_000)
    return path

@pytest.fixture(scope="session")
def fake_user():
    return "42"
