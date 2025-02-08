import sys
import os
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from types import SimpleNamespace
import json
import numpy as np


# Указать путь к репозиторию HiFi-GAN
hifi_gan_path = "D:/prhfg/hifi-gan"
sys.path.append(hifi_gan_path)

from models import Generator  # Импортируем Generator из локального HiFi-GAN

# Функция для загрузки модели HiFi-GAN
def load_model(checkpoint_path, config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = SimpleNamespace(**config_dict)
    generator = Generator(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator, config

# Функция для преобразования аудио в мел-спектрограмму
def audio_to_mel(audio_path, config):
    waveform, sr = torchaudio.load(audio_path)
    if sr != config.sampling_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=config.sampling_rate)
        waveform = resampler(waveform)
    mel_transform = T.MelSpectrogram(
        sample_rate=config.sampling_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_size,
        n_mels=config.num_mels,
        f_min=config.fmin,
        f_max=config.fmax,
    )
    mel = mel_transform(waveform)
    return mel

def save_audio(output_path, audio, samplerate):
    audio_np = audio.cpu().numpy()
    if np.isnan(audio_np).any() or np.isinf(audio_np).any():
        raise ValueError("Аудиоданные содержат некорректные значения (NaN или Inf).")
    sf.write(output_path, audio_np, samplerate)

# Функция для улучшения аудио
def enhance_audio(input_audio, output_audio, model_path, config_path):
    print("Загрузка модели и конфигурации...")
    generator, config = load_model(model_path, config_path)

    print("Преобразование аудио в мел-спектрограмму...")
    mel = audio_to_mel(input_audio, config)

    print("Генерация улучшенного аудио...")
    with torch.no_grad():
        mel = mel.squeeze(0)  # Убираем лишнее измерение
        audio = generator(mel.unsqueeze(0)).squeeze(0)

    print("Сохранение улучшенного аудио...")
    save_audio(output_audio, audio, config.sampling_rate)
    print(f"Улучшенное аудио сохранено в {output_audio}")

# Запуск процесса
model_path = "D:/prhfg/LJ_FT_T2_V2/generator"
config_path = "D:/prhfg/LJ_FT_T2_V2/config.json"
input_audio = "D:/prhfg/p_32896185_362.wav"
output_audio = "D:/prhfg/output_fixed.wav"

enhance_audio(input_audio, output_audio, model_path, config_path)
