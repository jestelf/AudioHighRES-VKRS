import sys
import os
import numpy as np
import torch
import torchaudio
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs
from PyQt5.QtWidgets import QApplication, QFileDialog

# Разрешаем безопасно загружать необходимые классы
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Создаем приложение PyQt5
app = QApplication(sys.argv)

# Диалог выбора файла голосового слепка
options = QFileDialog.Options()
options |= QFileDialog.ReadOnly
embedding_file, _ = QFileDialog.getOpenFileName(
    None,
    "Выберите файл голосового слепка", 
    "",
    "NPZ Files (*.npz);;All Files (*)", 
    options=options
)
if not embedding_file:
    print("Файл не выбран.")
    sys.exit()

print("Используем файл голосового слепка:", embedding_file)

# Инициализация модели XTTSv2 (с gpu=True)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Получаем устройство из параметров модели
device = next(tts.synthesizer.tts_model.parameters()).device
tts.to(device)

# Принудительно переводим модель в float32
tts.synthesizer.tts_model = tts.synthesizer.tts_model.float()

# Загружаем эмбеддинги из NPZ и явно приводим к float32
data = np.load(embedding_file)
gpt_cond_latent_np = data["gpt_cond_latent"]
speaker_embedding_np = data["speaker_embedding"]

gpt_cond_latent = torch.tensor(gpt_cond_latent_np, dtype=torch.float32, device=device)
speaker_embedding = torch.tensor(speaker_embedding_np, dtype=torch.float32, device=device)

# Текст для синтеза
text = "Hello, my name is Genry! You`re a welcome!"

# Отключаем AMP с использованием нового синтаксиса (для cuda) – чтобы избежать преобразования в half
if device.type == "cuda":
    autocast_context = torch.amp.autocast(device_type="cuda", enabled=False)
else:
    autocast_context = torch.amp.autocast(enabled=False)

with autocast_context:
    out = tts.synthesizer.tts_model.inference(
        text, 
        "ru", 
        gpt_cond_latent, 
        speaker_embedding, 
        temperature=0.7
    )

# Получаем аудио (ключ "wav") и убеждаемся, что это тензор float32
wav = out["wav"]
if not isinstance(wav, torch.Tensor):
    wav = torch.tensor(wav, dtype=torch.float32, device=device)
else:
    wav = wav.float()

# Если тензор одномерный (samples,), добавляем размер канала
if wav.ndim == 1:
    wav = wav.unsqueeze(0)

# Сохраняем аудио в WAV-файл (частота 24000 Гц)
output_audio = "cloned_voice.wav"
torchaudio.save(output_audio, wav.cpu(), 24000)
print("Аудио сохранено в", output_audio)
