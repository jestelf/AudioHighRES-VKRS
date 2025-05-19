import sys
import os
import numpy as np
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs
from PyQt5.QtWidgets import QApplication, QFileDialog
from datetime import datetime
from pydub import AudioSegment

# Разрешаем безопасно загружать необходимые классы
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Создаем приложение PyQt5
app = QApplication(sys.argv)

# Диалог выбора аудиофайла
options = QFileDialog.Options()
options |= QFileDialog.ReadOnly
fileName, _ = QFileDialog.getOpenFileName(None,
    "Выберите аудиофайл для создания голосового слепка", "",
    "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)", options=options)
if not fileName:
    print("Файл не выбран.")
    sys.exit()

# Если файл не в формате wav, выполняем конвертацию
base, ext = os.path.splitext(fileName)
if ext.lower() != ".wav":
    print("Конвертация файла в WAV...")
    audio = AudioSegment.from_file(fileName)
    wav_file = base + ".wav"
    audio.export(wav_file, format="wav")
    file_to_use = wav_file
else:
    file_to_use = fileName

print("Используем файл:", file_to_use)

# Инициализация модели XTTSv2
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Вычисляем эмбеддинги для голосового клонирования
gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(audio_path=[file_to_use])

# Переносим тензоры на CPU и преобразуем в NumPy массивы
gpt_cond_latent_cpu = gpt_cond_latent.cpu().numpy()
speaker_embedding_cpu = speaker_embedding.cpu().numpy()

# Генерируем имя выходного файла с уникальным номером (на основе метки времени)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"speaker_embedding_{timestamp}.npz"
np.savez(output_file, gpt_cond_latent=gpt_cond_latent_cpu, speaker_embedding=speaker_embedding_cpu)
print("Голосовой слепок сохранён в", output_file)
