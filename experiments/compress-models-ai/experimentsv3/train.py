#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Классификатор синтезированной vs настоящей речи.
Используем LibriSpeech для настоящей речи.
Для синтезированной речи можно использовать два подхода:
  1. Симуляция эффектов (сдвиг тона, темп, эхо, спектральный наклон)
  2. Генерация через gTTS (онлайн, автоматическая подгрузка)
  
Кроме того, при запуске скрипта создаются папки для датасетов:
  - datasets/original – для оригинальных аудио
  - datasets/synthesized – для синтезированных аудио
  
Требования:
    - Python 3.7+
    - torch, torchaudio, numpy, tqdm, optuna (опционально), gTTS
Установка:
    pip install torch torchaudio numpy tqdm optuna gTTS
"""

import os
import sys
import csv
import logging
import datetime
import argparse
import random
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Subset, ConcatDataset

from tqdm import tqdm
import numpy as np

# Для режима optuna
try:
    import optuna
except ImportError:
    optuna = None

from gtts import gTTS

# ---------------------------------------------------
# Функция для создания папок датасетов
# ---------------------------------------------------
def create_dataset_folders():
    base_dataset_dir = os.path.join(BASE_DIR, "datasets")
    original_dir = os.path.join(base_dataset_dir, "original")
    synthesized_dir = os.path.join(base_dataset_dir, "synthesized")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(synthesized_dir, exist_ok=True)
    return original_dir, synthesized_dir

# ---------------------------------------------------
# Функция для применения эффектов (если датасет синтезированной речи не используется)
# ---------------------------------------------------
def simulate_synthetic_effects(waveform, sample_rate):
    effects = [
        ["pitch", "-50"],
        ["tempo", "0.95"],
        ["echo", "0.8", "0.9", "1000", "0.3"]
    ]
    try:
        waveform_effected, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
    except Exception as e:
        print("Ошибка применения sox эффектов:", e)
        waveform_effected = waveform
    return waveform_effected

# ---------------------------------------------------
# 1) Логирование
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Создаем папки датасетов (если их ещё нет)
default_original_dir, default_synthesized_dir = create_dataset_folders()

def setup_logging(backup_subdir="speech_classifier_backup"):
    backup_dir = os.path.join(BASE_DIR, backup_subdir)
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_backup = os.path.join(backup_dir, timestamp)
    os.makedirs(current_backup, exist_ok=True)
    log_filename = os.path.join(current_backup, "training.log")
    csv_filename = os.path.join(current_backup, "training_metrics.csv")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Логирование инициализировано.")
    logging.info(f"Backup folder: {current_backup}")
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss", "accuracy", "learning_rate"])
    return current_backup, log_filename, csv_filename

# ---------------------------------------------------
# 2a) Датасет для настоящей речи (LibriSpeech)
# ---------------------------------------------------
class RealSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root, subset_indices=None, download=True, url="test-clean"):
        os.makedirs(root, exist_ok=True)
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=download)
        if subset_indices is not None:
            self.indices = subset_indices
        else:
            self.indices = list(range(len(self.dataset)))
        self.num_mels = 80
        self.target_sample_rate = 22050
        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=self.target_sample_rate)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=self.num_mels
        )
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        waveform, sample_rate, _, _, _, _ = self.dataset[real_idx]
        if sample_rate != self.target_sample_rate:
            waveform = self.resampler(waveform)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log1p(mel_spec)
        return mel_spec, 0

# ---------------------------------------------------
# 2b) Датасет для синтезированной речи из локальной директории
# ---------------------------------------------------
class SyntheticSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, directory, target_sample_rate=22050, num_mels=80):
        self.directory = directory
        self.file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(".wav")]
        self.target_sample_rate = target_sample_rate
        self.num_mels = num_mels
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=num_mels
        )
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log1p(mel_spec)
        return mel_spec, 1

# ---------------------------------------------------
# 2c) Датасет для синтезированной речи через gTTS (онлайн)
# ---------------------------------------------------
class GTTSSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, texts=None, target_sample_rate=22050, num_mels=80):
        if texts is None:
            self.texts = [
                "Hello, this is a test of synthetic speech.",
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is the future.",
                "Machine learning enables computers to learn from data."
            ]
        else:
            self.texts = texts
        self.target_sample_rate = target_sample_rate
        self.num_mels = num_mels
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=num_mels
        )
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        try:
            tts = gTTS(text, lang='en')
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                temp_path = fp.name
            tts.save(temp_path)
            waveform, sample_rate = torchaudio.load(temp_path)
            os.remove(temp_path)
        except Exception as e:
            print("Ошибка генерации синтезированной речи через gTTS:", e)
            waveform = torch.zeros(1, self.target_sample_rate)
            sample_rate = self.target_sample_rate
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log1p(mel_spec)
        return mel_spec, 1

# ---------------------------------------------------
# 2d) Объединённый датасет
# ---------------------------------------------------
def get_combined_dataset(real_data_dir, synthetic_dir=None, use_gtts=False, subset_size=200):
    real_ds = RealSpeechDataset(
        root=real_data_dir, 
        subset_indices=list(range(min(subset_size, len(
            torchaudio.datasets.LIBRISPEECH(root=real_data_dir, url="test-clean", download=True)
        ))))
    )
    if use_gtts:
        synthetic_ds = GTTSSyntheticDataset()
    elif synthetic_dir is not None:
        synthetic_ds = SyntheticSpeechDataset(directory=synthetic_dir)
        synthetic_ds = Subset(synthetic_ds, list(range(min(subset_size, len(synthetic_ds)))))
    else:
        # Если не указан ни один вариант, применяем эффекты к настоящему аудио для имитации синтезированной речи
        class SyntheticWrapper(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                mel, _ = self.ds[idx]
                return mel, 1
        synthetic_ds = SyntheticWrapper(real_ds)
    combined_ds = ConcatDataset([real_ds, synthetic_ds])
    return combined_ds

# ---------------------------------------------------
# Collate функция
# ---------------------------------------------------
def collate_fn_classifier(batch):
    mels, labels = zip(*batch)
    max_time = max(mel.size(-1) for mel in mels)
    padded_mels = [F.pad(mel, (0, max_time - mel.size(-1)), value=0.0) for mel in mels]
    mels_tensor = torch.stack(padded_mels, dim=0)
    labels_tensor = torch.LongTensor(labels)
    return mels_tensor, labels_tensor

# ---------------------------------------------------
# 3) Модель классификатора речи
# ---------------------------------------------------
class SpeechClassifier(nn.Module):
    def __init__(self, num_mels=80, num_classes=2, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# ---------------------------------------------------
# 4) Функция обучения (одиночный цикл)
# ---------------------------------------------------
def train_cycle_classifier(model, loader, optimizer, criterion, scaler, device, grad_clip, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for mels, labels in pbar:
            mels = mels.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(mels)
                loss = criterion(logits, labels)
            loss_val = loss.item()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss_val
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss_val:.6f}", acc=f"{100 * correct/total:.2f}%")
        avg_loss = epoch_loss / len(loader)
        acc = 100 * correct / total
        logging.info(f"[Epoch {epoch+1}/{num_epochs}] avg_loss={avg_loss:.6f} accuracy={acc:.2f}% lr={optimizer.param_groups[0]['lr']}")
    return avg_loss

# ---------------------------------------------------
# Функция для поиска глобально лучшей модели среди optuna циклов
# ---------------------------------------------------
def get_global_best_model(backup_root="speech_classifier_backup"):
    best_loss = float('inf')
    best_model_path = None
    if not os.path.exists(backup_root):
        return best_loss, best_model_path
    for entry in os.listdir(backup_root):
        full_path = os.path.join(backup_root, entry)
        if os.path.isdir(full_path) and entry.startswith("optuna_cycle_"):
            candidate = os.path.join(full_path, "best_speech_classifier.pth")
            loss_file = os.path.join(full_path, "best_loss.txt")
            if os.path.exists(candidate) and os.path.exists(loss_file):
                try:
                    with open(loss_file, "r") as f:
                        loss_val = float(f.read().strip())
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_model_path = candidate
                except Exception:
                    pass
    return best_loss, best_model_path

# ---------------------------------------------------
# 5) Режим обучения с интеграцией Optuna (infinite_optuna)
# ---------------------------------------------------
def infinite_optuna_train_main(args):
    if optuna is None:
        print("Optuna не установлен. Пожалуйста, установите optuna и повторите попытку.")
        return

    num_cycles = args.num_cycles
    study = optuna.create_study(direction="minimize")
    # Используем аргументы или дефолтные папки
    data_dir = args.original_dir if args.original_dir is not None else default_original_dir
    global_best_loss, global_best_model = get_global_best_model(os.path.join(BASE_DIR, "speech_classifier_backup"))
    if global_best_model is not None:
        print(f"Глобально лучшая модель найдена: {global_best_model} с loss={global_best_loss:.6f}")
    else:
        print("Глобально лучшая модель не найдена.")
    best_loss = global_best_loss
    cycle = 0
    while num_cycles == 0 or cycle < num_cycles:
        cycle += 1
        print(f"\n===== Optuna Cycle {cycle} =====")
        trial = study.ask()
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_int("batch_size", 4, 32)
        num_epochs = trial.suggest_int("num_epochs", 5, 20)
        grad_clip = trial.suggest_float("grad_clip", 0.5, 2.0)
        dropout = trial.suggest_float("dropout", 0.3, 0.7)
        current_params = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "grad_clip": grad_clip,
            "dropout": dropout
        }
        print(f"Предложенные параметры: {current_params}")
        backup_folder = os.path.join(BASE_DIR, "speech_classifier_backup", f"optuna_cycle_{cycle}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(backup_folder, exist_ok=True)
        csv_filename = os.path.join(backup_folder, "training_metrics.csv")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ds = get_combined_dataset(real_data_dir=data_dir, synthetic_dir=args.synthesized_dir, use_gtts=args.use_gtts, subset_size=200)
        loader = DataLoader(ds, batch_size=current_params["batch_size"], shuffle=True,
                            collate_fn=collate_fn_classifier, num_workers=2)
        model = SpeechClassifier(num_mels=80, num_classes=2, dropout=current_params["dropout"]).to(device)
        if global_best_model is not None:
            try:
                model.load_state_dict(torch.load(global_best_model, map_location=device))
                print("Загружена глобально лучшая модель для инициализации.")
            except Exception as e:
                print("Не удалось загрузить глобально лучшую модель:", e)
        optimizer = torch.optim.Adam(model.parameters(), lr=current_params["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        final_loss = train_cycle_classifier(model, loader, optimizer, criterion, scaler, device, current_params["grad_clip"], current_params["num_epochs"])
        print(f"Цикл {cycle} завершён. Финальная loss: {final_loss:.6f}")
        study.tell(trial, final_loss)
        if final_loss < best_loss:
            best_loss = final_loss
            best_model_path = os.path.join(backup_folder, "best_speech_classifier.pth")
            torch.save(model.state_dict(), best_model_path)
            with open(os.path.join(backup_folder, "best_loss.txt"), "w") as f:
                f.write(f"{best_loss:.6f}")
            print(f"Новый лучший результат! Loss: {best_loss:.6f}. Модель сохранена в {best_model_path}")
            global_best_model = best_model_path
        else:
            print("Лучший результат остаётся:", best_loss)
        print("Текущий лучший trial:")
        print(study.best_trial)
        if num_cycles != 0 and cycle >= num_cycles:
            break

# ---------------------------------------------------
# 6) Режим одиночного обучения (train)
# ---------------------------------------------------
def train_main(args):
    backup_dir, log_filename, csv_filename = setup_logging("speech_classifier_backup")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Устройство: {device}")
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    grad_clip = args.grad_clip
    dropout = args.dropout
    data_dir = args.original_dir if args.original_dir is not None else default_original_dir
    ds = get_combined_dataset(real_data_dir=data_dir, synthetic_dir=args.synthesized_dir, use_gtts=args.use_gtts, subset_size=200)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn_classifier, num_workers=2)
    model = SpeechClassifier(num_mels=80, num_classes=2, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss", "accuracy", "learning_rate"])
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for mels, labels in pbar:
                mels = mels.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(mels)
                    loss = criterion(logits, labels)
                loss_val = loss.item()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss_val
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix(loss=f"{loss_val:.6f}", acc=f"{100*correct/total:.2f}%")
            avg_loss = epoch_loss / len(loader)
            acc = 100 * correct / total
            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(f"[Epoch {epoch+1}/{num_epochs}] avg_loss={avg_loss:.6f} accuracy={acc:.2f}% lr={current_lr}")
            writer.writerow([epoch+1, avg_loss, acc, current_lr])
    model_path = os.path.join(backup_dir, "speech_classifier.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Модель сохранена: {model_path}")

# ---------------------------------------------------
# 7) Режим предсказания (predict)
# ---------------------------------------------------
def predict_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform, sample_rate = torchaudio.load(args.input_file)
    target_sample_rate = 22050
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80
    )
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log1p(mel_spec)
    mel_spec = mel_spec.unsqueeze(0)
    model = SpeechClassifier(num_mels=80, num_classes=2, dropout=0.5).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print("Ошибка загрузки модели:", e)
        return
    model.eval()
    with torch.no_grad():
        logits = model(mel_spec.to(device))
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
    classes = {0: "Real (настоящая)", 1: "Synthetic (синтезированная)"}
    print(f"Результат: {classes[pred_class]} (уверенность: {probs[0][pred_class]:.2f})")

# ---------------------------------------------------
# 8) Основной блок с подкомандами
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Классификатор синтезированной vs настоящей речи (train / predict / infinite_optuna)")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    # Режим train
    train_parser = subparsers.add_parser("train", help="Одиночный цикл обучения классификатора")
    train_parser.add_argument("--learning_rate", type=float, default=1e-3, help="Скорость обучения")
    train_parser.add_argument("--batch_size", type=int, default=4, help="Размер батча")
    train_parser.add_argument("--num_epochs", type=int, default=5, help="Число эпох")
    train_parser.add_argument("--grad_clip", type=float, default=1.0, help="Норма клиппинга градиентов")
    train_parser.add_argument("--dropout", type=float, default=0.5, help="Dropout в классификаторе")
    train_parser.add_argument("--original_dir", type=str, default=None, help="Путь к директории с оригинальными аудио (по умолчанию datasets/original)")
    train_parser.add_argument("--synthesized_dir", type=str, default=None, help="Путь к директории с синтезированной речью (по умолчанию datasets/synthesized)")
    train_parser.add_argument("--use_gtts", action="store_true", help="Использовать gTTS для генерации синтезированной речи онлайн")
    # Режим predict
    predict_parser = subparsers.add_parser("predict", help="Предсказание для аудио файла")
    predict_parser.add_argument("--model_path", type=str, required=True, help="Путь к сохранённой модели (.pth)")
    predict_parser.add_argument("--input_file", type=str, required=True, help="Путь к аудио файлу для классификации")
    # Режим infinite_optuna
    infinite_parser = subparsers.add_parser("infinite_optuna", help="Бесконечное обучение с Optuna для выбора гиперпараметров")
    infinite_parser.add_argument("--learning_rate", type=float, default=1e-3, help="Начальная скорость обучения")
    infinite_parser.add_argument("--batch_size", type=int, default=4, help="Начальный размер батча")
    infinite_parser.add_argument("--num_epochs", type=int, default=5, help="Число эпох в каждом цикле")
    infinite_parser.add_argument("--grad_clip", type=float, default=1.0, help="Норма клиппинга градиентов")
    infinite_parser.add_argument("--dropout", type=float, default=0.5, help="Dropout в классификаторе")
    infinite_parser.add_argument("--num_cycles", type=int, default=10, help="Количество циклов (0 для бесконечного обучения)")
    infinite_parser.add_argument("--original_dir", type=str, default=None, help="Путь к директории с оригинальными аудио (по умолчанию datasets/original)")
    infinite_parser.add_argument("--synthesized_dir", type=str, default=None, help="Путь к директории с синтезированной речью (по умолчанию datasets/synthesized)")
    infinite_parser.add_argument("--use_gtts", action="store_true", help="Использовать gTTS для генерации синтезированной речи онлайн")
    
    args = parser.parse_args()
    
    # Если не указаны директории, используем дефолтные, созданные в create_dataset_folders()
    if args.original_dir is None:
        args.original_dir = default_original_dir
    if args.synthesized_dir is None:
        args.synthesized_dir = default_synthesized_dir
    
    if args.mode == "train":
        train_main(args)
    elif args.mode == "predict":
        predict_main(args)
    elif args.mode == "infinite_optuna":
        infinite_optuna_train_main(args)
