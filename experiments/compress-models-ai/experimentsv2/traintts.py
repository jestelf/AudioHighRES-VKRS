#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import logging
import datetime
import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
import numpy as np

# Для режима optuna
try:
    import optuna
except ImportError:
    optuna = None

# ---------------------------------------------------
# 1) Логирование
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_logging(backup_subdir="ms_tts_backup"):
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
        writer.writerow(["epoch", "avg_loss", "delta", "learning_rate"])

    return current_backup, log_filename, csv_filename

# ---------------------------------------------------
# 2) Датасет Multi-speaker на основе LibriSpeech
# ---------------------------------------------------
class LibriSpeechMultiSpeakerDataset(torch.utils.data.Dataset):
    """
    Датасет на основе torchaudio.datasets.LIBRISPEECH.
    Каждый элемент содержит:
      - waveform, sample_rate
      - текст (транскрипция)
      - speaker_id
    Генерируются:
      - text_tokens (упрощённая токенизация)
      - мел-спектрограмма
      - speaker_id (int)
    """
    def __init__(self, root, subset_indices=None, download=True, url="test-clean"):
        os.makedirs(root, exist_ok=True)
        
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=url, download=download
        )
        if subset_indices is not None:
            self.indices = subset_indices
        else:
            self.indices = list(range(len(self.dataset)))
        
        self.char2idx = {}
        valid_chars = " abcdefghijklmnopqrstuvwxyz0123456789.,!?'-"
        for i, ch in enumerate(valid_chars):
            self.char2idx[ch] = i + 1
        self.vocab_size = len(self.char2idx) + 1  # +1 для PAD=0
        
        self.num_mels = 80
        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=self.num_mels
        )
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        waveform, sample_rate, utterance, speaker_id, _, _ = self.dataset[real_idx]
        if sample_rate != 22050:
            waveform = self.resampler(waveform)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        text_str = utterance.lower()
        text_tokens = [self.char2idx[ch] for ch in text_str if ch in self.char2idx]
        text_tokens = torch.LongTensor(text_tokens)
        mel_spec = self.mel_transform(waveform)  # (1, num_mels, time)
        mel_spec = torch.log1p(mel_spec)
        mel_spec = mel_spec.squeeze(0)  # -> (num_mels, time)
        return text_tokens, mel_spec, speaker_id

def collate_fn_ms(batch):
    texts, mels, spk_ids = zip(*batch)
    max_text_len = max(t.size(0) for t in texts)
    max_mel_len = max(m.size(-1) for m in mels)
    padded_texts = [F.pad(t, (0, max_text_len - t.size(0)), value=0) for t in texts]
    padded_mels = [F.pad(m, (0, max_mel_len - m.size(-1)), value=0.0) for m in mels]
    text_tensor = torch.stack(padded_texts, dim=0)
    mel_tensor = torch.stack(padded_mels, dim=0)
    speaker_tensor = torch.LongTensor(spk_ids)
    return text_tensor, mel_tensor, speaker_tensor

# ---------------------------------------------------
# 3) Модель Multi-speaker TTS (упрощённая)
# ---------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
    
    def forward(self, text_tokens):
        x = self.embedding(text_tokens)  # (B, T, embed_dim)
        out, _ = self.lstm(x)            # (B, T, 2*embed_dim)
        return out

class MultiSpeakerTTS(nn.Module):
    def __init__(self, vocab_size, text_embed_dim=128, spk_embed_dim=64, num_speakers=2000, num_mels=80):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, text_embed_dim)
        self.speaker_table = nn.Embedding(num_speakers, spk_embed_dim)
        input_dim = 2 * text_embed_dim + spk_embed_dim
        self.gru = nn.GRU(input_dim, 256, batch_first=True)
        self.linear = nn.Linear(256, num_mels)
    
    def forward(self, text_tokens, speaker_id=None, speaker_imprint=None):
        B, T = text_tokens.shape
        text_feats = self.text_encoder(text_tokens)  # (B, T, 2*text_embed_dim)
        if speaker_imprint is not None:
            spk_vec = speaker_imprint.unsqueeze(1).expand(-1, T, -1)
        else:
            spk_emb = self.speaker_table(speaker_id)  # (B, spk_embed_dim)
            spk_vec = spk_emb.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([text_feats, spk_vec], dim=2)
        out, _ = self.gru(x)
        mel = self.linear(out)  # (B, T, num_mels)
        return mel

# ---------------------------------------------------
# 4) Функция обучения (одиночный цикл)
# ---------------------------------------------------
def train_cycle(model, loader, optimizer, criterion, scaler, device, grad_clip, num_epochs):
    epoch_loss_total = 0.0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for text_tokens, mel_specs, spk_ids in pbar:
            text_tokens = text_tokens.to(device)
            mel_specs = mel_specs.to(device)
            spk_ids = spk_ids.to(device)
            T_text = text_tokens.size(1)
            T_mel = mel_specs.size(2)
            max_len = min(T_text, T_mel)
            text_tokens = text_tokens[:, :max_len]
            mel_specs = mel_specs[:, :, :max_len].transpose(1, 2)  # (B, T, num_mels)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == "cuda")):
                pred_mel = model(text_tokens, speaker_id=spk_ids, speaker_imprint=None)
                loss = criterion(pred_mel, mel_specs)
            loss_val = loss.item()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss_val
            pbar.set_postfix(loss=f"{loss_val:.6f}")
        avg_loss = epoch_loss / len(loader)
        epoch_loss_total += avg_loss
    final_loss = epoch_loss_total / num_epochs
    return final_loss

# ---------------------------------------------------
# Функция для поиска глобально лучшей модели среди всех optuna_cycle папок
# ---------------------------------------------------
def get_global_best_model(backup_root="ms_tts_backup"):
    best_loss = float('inf')
    best_model_path = None
    if not os.path.exists(backup_root):
        return best_loss, best_model_path
    for entry in os.listdir(backup_root):
        full_path = os.path.join(backup_root, entry)
        if os.path.isdir(full_path) and entry.startswith("optuna_cycle_"):
            candidate = os.path.join(full_path, "best_ms_tts.pth")
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
# 5) Режим бесконечного обучения с интеграцией Optuna и сравнением с глобальным результатом (infinite_optuna)
# ---------------------------------------------------
def infinite_optuna_train_main(args):
    if optuna is None:
        print("Optuna не установлен. Пожалуйста, установите optuna и повторите попытку.")
        return

    num_cycles = args.num_cycles  # если 0, то бесконечно
    study = optuna.create_study(direction="minimize")
    data_dir = os.path.join(BASE_DIR, "data_librispeech")
    global_best_loss, global_best_model = get_global_best_model(os.path.join(BASE_DIR, "ms_tts_backup"))
    if global_best_model is not None:
        print(f"Глобально лучшая модель найдена: {global_best_model} с loss={global_best_loss:.6f}")
    else:
        print("Глобально лучшая модель не найдена.")
    best_loss = global_best_loss  # используем глобальный лучший, если есть
    cycle = 0
    while num_cycles == 0 or cycle < num_cycles:
        cycle += 1
        print(f"\n===== Optuna Cycle {cycle} =====")
        trial = study.ask()
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_int("batch_size", 4, 32)
        num_epochs = trial.suggest_int("num_epochs", 5, 20)
        grad_clip = trial.suggest_float("grad_clip", 0.5, 2.0)
        current_params = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "grad_clip": grad_clip
        }
        print(f"Предложенные параметры: {current_params}")

        backup_folder = os.path.join(BASE_DIR, "ms_tts_backup", f"optuna_cycle_{cycle}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(backup_folder, exist_ok=True)
        csv_filename = os.path.join(backup_folder, "training_metrics.csv")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ds = LibriSpeechMultiSpeakerDataset(root=data_dir, download=True, url="test-clean")
        subset_indices = list(range(min(200, len(ds))))
        ds_small = Subset(ds, subset_indices)
        loader = DataLoader(ds_small, batch_size=current_params["batch_size"], shuffle=True,
                            collate_fn=collate_fn_ms, num_workers=2)

        model = MultiSpeakerTTS(vocab_size=ds.vocab_size,
                                text_embed_dim=128,
                                spk_embed_dim=64,
                                num_speakers=2000,
                                num_mels=80).to(device)
        # Если глобально лучшая модель существует, загружаем её как начальное состояние
        if global_best_model is not None:
            try:
                model.load_state_dict(torch.load(global_best_model, map_location=device))
                print("Загружена глобально лучшая модель для инициализации.")
            except Exception as e:
                print("Не удалось загрузить глобально лучшую модель:", e)
        optimizer = torch.optim.Adam(model.parameters(), lr=current_params["learning_rate"])
        criterion = nn.L1Loss()
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        
        final_loss = train_cycle(model, loader, optimizer, criterion, scaler, device, current_params["grad_clip"], current_params["num_epochs"])
        print(f"Цикл {cycle} завершён. Финальная loss: {final_loss:.6f}")
        study.tell(trial, final_loss)
        # Если полученная loss лучше, обновляем глобальный лучший результат и сохраняем модель
        if final_loss < best_loss:
            best_loss = final_loss
            best_model_path = os.path.join(backup_folder, "best_ms_tts.pth")
            torch.save(model.state_dict(), best_model_path)
            # Сохраняем также файл с лучшей loss
            with open(os.path.join(backup_folder, "best_loss.txt"), "w") as f:
                f.write(f"{best_loss:.6f}")
            print(f"Новый лучший результат! Loss: {best_loss:.6f}. Модель сохранена в {best_model_path}")
            # Обновляем глобальный лучший модель, чтобы следующие циклы использовали её
            global_best_model = best_model_path
        else:
            print("Лучший результат остаётся:", best_loss)
        print("Текущий лучший trial:")
        print(study.best_trial)
        if num_cycles != 0 and cycle >= num_cycles:
            break

# ---------------------------------------------------
# 6) Режим одиночного обучения (train) и синтеза (synthesize)
# ---------------------------------------------------
def train_main(args):
    backup_dir, log_filename, csv_filename = setup_logging("ms_tts_backup")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Устройство: {device}")

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    grad_clip = args.grad_clip

    data_dir = os.path.join(BASE_DIR, "data_librispeech")
    ds = LibriSpeechMultiSpeakerDataset(root=data_dir, download=True, url="test-clean")
    subset_indices = list(range(min(200, len(ds))))
    ds_small = Subset(ds, subset_indices)
    loader = DataLoader(ds_small, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn_ms, num_workers=2)

    model = MultiSpeakerTTS(vocab_size=ds.vocab_size,
                            text_embed_dim=128,
                            spk_embed_dim=64,
                            num_speakers=2000,
                            num_mels=80).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    prev_epoch_loss = None
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss", "delta", "learning_rate"])
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for text_tokens, mel_specs, spk_ids in pbar:
                text_tokens = text_tokens.to(device)
                mel_specs = mel_specs.to(device)
                spk_ids = spk_ids.to(device)
                T_text = text_tokens.size(1)
                T_mel = mel_specs.size(2)
                max_len = min(T_text, T_mel)
                text_tokens = text_tokens[:, :max_len]
                mel_specs = mel_specs[:, :, :max_len].transpose(1, 2)  # (B, T, num_mels)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', enabled=(device.type == "cuda")):
                    pred_mel = model(text_tokens, speaker_id=spk_ids, speaker_imprint=None)
                    loss = criterion(pred_mel, mel_specs)
                loss_val = loss.item()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss_val
                pbar.set_postfix(loss=f"{loss_val:.6f}")
            avg_loss = epoch_loss / len(loader)
            current_lr = optimizer.param_groups[0]["lr"]
            delta = (prev_epoch_loss - avg_loss) if prev_epoch_loss is not None else 0.0
            logging.info(f"[Epoch {epoch+1}/{num_epochs}] avg_loss={avg_loss:.6f} delta={delta:.6f} lr={current_lr}")
            writer.writerow([epoch+1, avg_loss, delta, current_lr])
            prev_epoch_loss = avg_loss

    model_path = os.path.join(backup_dir, "ms_tts.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Модель сохранена: {model_path}")

def synthesize_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_ds = LibriSpeechMultiSpeakerDataset(root=os.path.join(BASE_DIR, "data_librispeech"),
                                               subset_indices=[0], download=False, url="test-clean")
    vocab_size = dummy_ds.vocab_size
    char2idx = dummy_ds.char2idx

    model = MultiSpeakerTTS(vocab_size=vocab_size,
                            text_embed_dim=128,
                            spk_embed_dim=64,
                            num_speakers=2000,
                            num_mels=80).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    text_to_speak = "Hello! This is a multi speaker T T S test."
    tokens = [char2idx[ch] for ch in text_to_speak.lower() if ch in char2idx]
    text_tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)

    if args.imprint_file is not None:
        data = np.load(args.imprint_file)
        if "imprint" not in data:
            print(f"Файл {args.imprint_file} не содержит ключ 'imprint'.")
            return
        imprint_np = data["imprint"]
        if imprint_np.shape[0] != 64:
            print(f"imprint.shape={imprint_np.shape}, а нужно (64,).")
            return
        imprint_tensor = torch.from_numpy(imprint_np).float().unsqueeze(0).to(device)
        speaker_id_tensor = None
        print("Используем ваш imprint вместо speaker_id.")
    else:
        sid = args.speaker_id if args.speaker_id is not None else 0
        speaker_id_tensor = torch.LongTensor([sid]).to(device)
        imprint_tensor = None
        print(f"Используем speaker_id={sid} из таблицы.")

    with torch.no_grad():
        pred_mel = model(text_tokens, speaker_id=speaker_id_tensor, speaker_imprint=imprint_tensor)
    pred_mel = torch.expm1(pred_mel[0].transpose(0, 1))

    inv_mel = torchaudio.transforms.MelInverse(n_fft=1024, n_mels=80, sample_rate=22050, n_iter=60)
    linear_spec = inv_mel(pred_mel.unsqueeze(0))
    waveform = linear_spec.squeeze(0).clamp(-1, 1)

    output_dir = os.path.join(BASE_DIR, "output_ms")
    os.makedirs(output_dir, exist_ok=True)
    out_wav = os.path.join(output_dir, "synthesized.wav")
    torchaudio.save(out_wav, waveform.unsqueeze(0), 22050)
    print(f"Синтез завершён: {out_wav}")

# ---------------------------------------------------
# 7) Основной блок с подкомандами
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Учебный Multi-Speaker TTS (train / synthesize / infinite_optuna)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Режим train
    train_parser = subparsers.add_parser("train", help="Одиночный цикл обучения модели TTS")
    train_parser.add_argument("--learning_rate", type=float, default=1e-3, help="Скорость обучения")
    train_parser.add_argument("--batch_size", type=int, default=4, help="Размер батча")
    train_parser.add_argument("--num_epochs", type=int, default=5, help="Число эпох")
    train_parser.add_argument("--grad_clip", type=float, default=1.0, help="Норма клиппинга градиентов")

    # Режим synthesize
    synth_parser = subparsers.add_parser("synthesize", help="Синтез речи с помощью обученной модели")
    synth_parser.add_argument("--model_path", type=str, required=True, help="Путь к ms_tts.pth")
    synth_parser.add_argument("--speaker_id", type=int, default=0, help="ID спикера (если не используем imprint)")
    synth_parser.add_argument("--imprint_file", type=str, default=None, help="NPZ с 'imprint' (spk_embed_dim,)")

    # Режим infinite_optuna
    infinite_parser = subparsers.add_parser("infinite_optuna", help="Бесконечное обучение с Optuna для выбора гиперпараметров")
    infinite_parser.add_argument("--learning_rate", type=float, default=1e-3, help="Начальная скорость обучения")
    infinite_parser.add_argument("--batch_size", type=int, default=4, help="Начальный размер батча")
    infinite_parser.add_argument("--num_epochs", type=int, default=5, help="Число эпох в каждом цикле")
    infinite_parser.add_argument("--grad_clip", type=float, default=1.0, help="Норма клиппинга градиентов")
    infinite_parser.add_argument("--num_cycles", type=int, default=10, help="Количество циклов (0 для бесконечного обучения)")

    args = parser.parse_args()

    if args.mode == "train":
        train_main(args)
    elif args.mode == "synthesize":
        synthesize_main(args)
    elif args.mode == "infinite_optuna":
        infinite_optuna_train_main(args)
