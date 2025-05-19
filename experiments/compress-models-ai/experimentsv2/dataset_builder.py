import os
import sys
import tarfile
import csv
import urllib.request
from datetime import datetime
import numpy as np
import torch
import torchaudio

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs

# Разрешаем безопасно загружать необходимые классы для весов
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Путь для датасета и URL
DATASET_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
DATASET_DIR = "LJSpeech-1.1"

def download_and_extract_dataset(url, target_dir):
    tar_file = os.path.basename(url)
    if not os.path.exists(tar_file):
        print(f"Скачиваем датасет из {url}...")
        urllib.request.urlretrieve(url, tar_file)
        print("Скачивание завершено.")
    else:
        print("Файл с датасетом уже скачан.")
    if not os.path.exists(target_dir):
        print(f"Распаковываем {tar_file}...")
        with tarfile.open(tar_file, "r:bz2") as tar:
            tar.extractall()
        print("Распаковка завершена.")
    else:
        print(f"Датасет уже распакован в {target_dir}.")

# Загружаем датасет, если его нет
download_and_extract_dataset(DATASET_URL, DATASET_DIR)

# Чтение метаданных (metadata.csv имеет формат: id|transcription|normalized transcription)
metadata_path = os.path.join(DATASET_DIR, "metadata.csv")
if not os.path.exists(metadata_path):
    print("Файл metadata.csv не найден!")
    sys.exit()

metadata = []
with open(metadata_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="|")
    for row in reader:
        if len(row) >= 2:
            metadata.append((row[0], row[1]))
print(f"Найдено {len(metadata)} записей в метаданных.")

# Обрабатываем весь датасет; если нужно ограничение, можно изменить N
N = len(metadata)
metadata_subset = metadata[:N]
print(f"Будет обработано {len(metadata_subset)} записей.")

# Инициализируем модель XTTSv2
print("Инициализация модели XTTSv2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
device = next(tts.synthesizer.tts_model.parameters()).device

# Папка для сохранения эмбеддингов и метаданных
output_folder = "embeddings_dataset"
os.makedirs(output_folder, exist_ok=True)
metadata_csv_path = os.path.join(output_folder, "embeddings_metadata.csv")
with open(metadata_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["serial", "audio_filename", "transcript", "embedding_filename"])

    serial = 1

    # Обрабатываем аудиофайлы
    for file_id, transcript in metadata_subset:
        audio_filename = os.path.join(DATASET_DIR, "wavs", f"{file_id}.wav")
        if not os.path.exists(audio_filename):
            print(f"Файл {audio_filename} не найден, пропускаем.")
            continue

        print(f"[{serial}] Обработка {audio_filename} ...")
        
        try:
            gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(audio_path=[audio_filename])
        except Exception as e:
            print(f"Ошибка при обработке {audio_filename}: {e}")
            continue

        gpt_cond_latent_cpu = gpt_cond_latent.cpu().numpy()
        speaker_embedding_cpu = speaker_embedding.cpu().numpy()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        embedding_filename = f"speaker_embedding_{serial:05d}_{timestamp}.npz"
        embedding_filepath = os.path.join(output_folder, embedding_filename)
        
        np.savez(embedding_filepath, gpt_cond_latent=gpt_cond_latent_cpu, speaker_embedding=speaker_embedding_cpu)
        print("Сохранён голосовой слепок в", embedding_filepath)

        csv_writer.writerow([serial, os.path.basename(audio_filename), transcript, embedding_filename])
        serial += 1

print("Создание датасета завершено!")
print("Метаданные сохранены в", metadata_csv_path)
