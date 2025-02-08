# tests/test_app.py
import pytest
from app import app
from io import BytesIO
import numpy as np
import soundfile as sf

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Audio Autoencoder Web Interface" in response.data

def test_compress_route(client):
    # Создаем dummy wav файл в памяти
    dummy_audio = (np.random.rand(16000) * 2 - 1).astype(np.float32)
    wav_io = BytesIO()
    sf.write(wav_io, dummy_audio, 16000, format='WAV')
    wav_io.seek(0)
    
    data = {
        "file": (wav_io, "dummy.wav"),
        "metadata": '{"sentence": "Тестовое аудио"}'
    }
    response = client.post("/compress", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    # Проверяем, что возвращается файл с расширением .npz
    assert response.headers["Content-Disposition"].find("compressed_file.npz") != -1

def test_decompress_route(client):
    # Создаем dummy npz файл в памяти
    dummy_latent = np.random.rand(1, 256)
    dummy_metadata = '{"sentence": "Тестовое аудио"}'
    npz_io = BytesIO()
    np.savez(npz_io, latent=dummy_latent, metadata=dummy_metadata)
    npz_io.seek(0)
    
    data = {
        "file": (npz_io, "dummy.npz")
    }
    response = client.post("/decompress", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    # Проверяем, что возвращается файл с расширением .wav
    assert response.headers["Content-Disposition"].find("reconstructed.wav") != -1
