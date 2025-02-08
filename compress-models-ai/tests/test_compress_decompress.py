# tests/test_compress_decompress.py
import torch
import numpy as np
from model import AudioAutoencoder
from compress_decompress import compress_audio, decompress_audio
import os

def test_compress_decompress(tmp_path):
    # Создаем временный dummy audio файл
    dummy_audio = (np.random.rand(16000) * 2 - 1).astype(np.float32)
    dummy_audio_path = tmp_path / "dummy.wav"
    import soundfile as sf
    sf.write(dummy_audio_path, dummy_audio, 16000)
    
    model = AudioAutoencoder()
    model.eval()
    latent = compress_audio(model, str(dummy_audio_path), segment_length=16000, device="cpu")
    assert latent is not None
    recon = decompress_audio(model, latent, segment_length=16000, device="cpu")
    assert recon.shape == (1, 16000)
