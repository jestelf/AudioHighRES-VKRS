import os
import glob
import shutil
import sys
import random
import tarfile
import urllib.request
import torch
from TTS.api import TTS
from datetime import datetime
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
import soundfile as sf  # для сохранения аудио

#############################################
# 1. Определение динамических путей относительно скрипта
#############################################
script_dir = os.path.dirname(os.path.abspath(__file__))
print("Script directory:", script_dir)

# Путь к датасету спикеров (ожидается, что папка находится рядом со скриптом)
english_dataset_path = os.path.join(script_dir, "english_speakers_dataset")
print("Путь к датасету спикеров:", english_dataset_path)

# Если папки нет, создаём её
if not os.path.exists(english_dataset_path):
    os.makedirs(english_dataset_path, exist_ok=True)
    print(f"Папка {english_dataset_path} создана.")

#############################################
# 2. Если нет подпапок, загружаем небольшой датасет Common Voice с Hugging Face
#############################################
speaker_dirs = [d for d in os.listdir(english_dataset_path)
                if os.path.isdir(os.path.join(english_dataset_path, d))]
if not speaker_dirs:
    print(f"Папка {english_dataset_path} существует, но не содержит подпапок со спикерами.")
    print("Попытка загрузки небольшого датасета Common Voice для английского языка...")

    try:
        # Загружаем 100 примеров из Common Voice (версии 11.0, конфигурация "en")
        try:
            cv_dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train[:100]")
        except Exception:
            cv_dataset = load_dataset("mozilla-foundation/common_voice", "en", split="train[:100]")
    except Exception as e:
        print("Ошибка при загрузке Common Voice:", e)
        sys.exit(1)

    # Группируем по client_id (идентификатору спикера)
    speakers = {}
    for sample in cv_dataset:
        cid = sample["client_id"]
        if cid not in speakers:
            speakers[cid] = []
        speakers[cid].append(sample)
    
    # Выбираем первых 3 спикеров (можно изменить число)
    selected_speaker_ids = list(speakers.keys())[:3]
    print("Будут использованы спикеры (client_id):", selected_speaker_ids)
    
    # Для каждого выбранного спикера создаём подпапку и сохраняем до 3 аудиофайлов
    for cid in selected_speaker_ids:
        speaker_folder = os.path.join(english_dataset_path, cid)
        os.makedirs(speaker_folder, exist_ok=True)
        samples = speakers[cid][:3]  # берем первые 3 аудиозаписи
        for idx, sample in enumerate(samples):
            # Поле "audio" содержит словарь с 'array' и 'sampling_rate'
            audio_array = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            out_filename = os.path.join(speaker_folder, f"sample_{idx+1}.wav")
            sf.write(out_filename, audio_array, sampling_rate)
            print(f"Сохранён файл {out_filename}")
    
    # Обновляем список подпапок
    speaker_dirs = [d for d in os.listdir(english_dataset_path)
                    if os.path.isdir(os.path.join(english_dataset_path, d))]
    if not speaker_dirs:
        raise ValueError("После загрузки датасета подпапок со спикерами не найдено.")

print("Найденные спикеры:", speaker_dirs)

# Для примера выбираем первого спикера
selected_speaker = speaker_dirs[0]
print("Выбран спикер:", selected_speaker)

#############################################
# 3. Создание папок для итогового датасета
#############################################
# Папка для итогового датасета: оригинальные аудио и синтезированные
dataset_folder = os.path.join(script_dir, "datasets")
original_folder = os.path.join(dataset_folder, "original")
synth_folder = os.path.join(dataset_folder, "synth")
os.makedirs(original_folder, exist_ok=True)
os.makedirs(synth_folder, exist_ok=True)

#############################################
# 4. Подготовка оригинальных аудио
#############################################
# Копируем все WAV-файлы выбранного спикера в папку original
speaker_audio_files = glob.glob(os.path.join(english_dataset_path, selected_speaker, "*.wav"))
if not speaker_audio_files:
    raise ValueError(f"В папке спикера {selected_speaker} не найдены WAV-файлы.")
print(f"Копирование {len(speaker_audio_files)} аудиофайлов в {original_folder}")
for audio_file in speaker_audio_files:
    shutil.copy(audio_file, original_folder)

#############################################
# 5. Инициализация TTS и получение эмбеддинга спикера
#############################################
# Инициализируем TTS (если GPU недоступно, установите gpu=False)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
audio_file_for_embedding = speaker_audio_files[0]
print("Используем аудиофайл для извлечения эмбеддинга:", audio_file_for_embedding)
with torch.no_grad():
    _, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(audio_path=[audio_file_for_embedding])
speaker_embedding_cpu = speaker_embedding.cpu().numpy()

#############################################
# 6. Синтез аудио по текстам из онлайн датасета
#############################################
print("Загружаем онлайн датасет с текстами (wikitext-2-raw-v1)...")
online_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
sentences = []
for entry in online_dataset:
    for line in entry["text"].split('\n'):
        line = line.strip()
        if line and len(line.split()) < 20:
            sentences.append(line)
    if len(sentences) >= 5:
        break
if not sentences:
    raise ValueError("Не удалось извлечь предложения из онлайн датасета.")
random.shuffle(sentences)
selected_sentences = sentences[:5]
print("Используем следующие предложения для синтеза:")
for idx, sentence in enumerate(selected_sentences):
    print(f"{idx+1}: {sentence}")

for idx, sentence in enumerate(selected_sentences):
    print(f"Синтез для предложения {idx+1}: {sentence}")
    synth_audio = tts.tts(text=sentence, speaker_embedding=speaker_embedding_cpu)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"synth_{idx+1}_{timestamp}.wav"
    output_path = os.path.join(synth_folder, output_filename)
    with open(output_path, "wb") as f:
        f.write(synth_audio)
    print("Сохранено синтезированное аудио в", output_path)

#############################################
# 7. Подготовка датасета для обучения классификатора
#############################################
SAMPLE_RATE = 16000  # желаемая частота дискретизации
FIXED_LENGTH = SAMPLE_RATE * 3  # фиксированная длина аудио (3 секунды)

def load_audio(file_path, sample_rate=SAMPLE_RATE, fixed_length=FIXED_LENGTH):
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    if waveform.size(1) > fixed_length:
        waveform = waveform[:, :fixed_length]
    else:
        pad_size = fixed_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    return waveform

original_files = glob.glob(os.path.join(original_folder, "*.wav"))
synth_files = glob.glob(os.path.join(synth_folder, "*.wav"))
all_files = original_files + synth_files
labels = [0] * len(original_files) + [1] * len(synth_files)
print(f"Общее число аудиофайлов для обучения: {len(all_files)}")

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        audio = load_audio(file_path)
        label = self.labels[idx]
        return audio, label

train_dataset = AudioDataset(all_files, labels)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

#############################################
# 8. Определение модели классификатора
#############################################
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        conv_output_length = FIXED_LENGTH // 4
        self.fc1 = nn.Linear(32 * conv_output_length, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = AudioClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#############################################
# 9. Обучение классификатора
#############################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

print("Начало обучения классификатора...")
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels_batch in train_loader:
        inputs = inputs.to(device)
        labels_batch = labels_batch.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Эпоха {epoch+1}/{num_epochs}, Потеря: {epoch_loss:.4f}")

print("Обучение завершено.")
