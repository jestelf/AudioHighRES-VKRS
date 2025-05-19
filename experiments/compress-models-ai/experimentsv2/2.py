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
embedding_file, _ = QFileDialog.getOpenFileName(None,
    "Выберите файл голосового слепка", "",
    "NPZ Files (*.npz);;All Files (*)", options=options)
if not embedding_file:
    print("Файл не выбран.")
    sys.exit()

print("Используем файл голосового слепка:", embedding_file)

# Инициализация модели XTTSv2
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Получаем устройство из параметров модели
device = next(tts.synthesizer.tts_model.parameters()).device

# Загружаем эмбеддинги из файла NPZ
data = np.load(embedding_file)
gpt_cond_latent_np = data["gpt_cond_latent"]
speaker_embedding_np = data["speaker_embedding"]

# Преобразуем массивы в тензоры и переносим на устройство модели
gpt_cond_latent = torch.tensor(gpt_cond_latent_np).to(device)
speaker_embedding = torch.tensor(speaker_embedding_np).to(device)

# Текст для синтеза (можно изменить по желанию)
text = "Привет, это тест голосового клонирования с использованием сохранённого слепка."

# Выполняем синтез речи с использованием загруженных эмбеддингов
out = tts.synthesizer.tts_model.inference(text, "ru", gpt_cond_latent, speaker_embedding, temperature=0.7)

# Сохраняем полученное аудио в файл
output_audio = "cloned_voice.wav"
torchaudio.save(output_audio, torch.tensor(out["wav"]).unsqueeze(0), 24000)
print("Аудио сохранено в", output_audio)
