#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение модели для извлечения голосового слепка (voice imprint).

Модель – сверточный автоэнкодер, который учится реконструировать аудио через сжатое 
латентное представление (слепок голоса). После обучения можно извлечь этот слепок 
(например, усреднением по времени) и сохранить его в сжатом формате (*.npz*).

Режимы работы (подкоманды):
  • train         : Одиночный цикл обучения.
  • extract       : Извлечение голосовых слепков с помощью сохранённой модели.
  • infinite      : Бесконечное обучение с изменением гиперпараметров (случайное обновление training/advanced).
  • optuna        : Оптимизация гиперпараметров с помощью Optuna (байесиановская оптимизация).
  • evolutionary  : Эволюционная (генетическая) оптимизация гиперпараметров с отбором, скрещиванием и мутацией.

Параметры передаются в три группы:
  [1] Training Parameters (некритичные): --learning_rate, --batch_size, --num_epochs, --grad_clip.
  [2] Advanced Parameters: --dropout_rate.
  [3] Architecture Parameters (критичные): --in_channels, --hidden_channels, --latent_dim, 
      --num_encoder_layers, --num_decoder_layers, --use_batch_norm.
      
Если в ходе обучения обнаруживается, что loss становится NaN, цикл прерывается (возвращается inf) и система переходит к следующей попытке.
"""

import os
import csv
import logging
import datetime
import argparse
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

# Для режима optuna:
try:
    import optuna
except ImportError:
    optuna = None

# Определяем базовую директорию – там, где лежит скрипт
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------- Настройка логирования -------------
def setup_logging(backup_subdir="backup"):
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
            logging.StreamHandler()
        ]
    )
    logging.info("Логирование инициализировано.")
    logging.info(f"Backup folder: {current_backup}")
    
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["epoch", "avg_loss", "delta", "learning_rate"])
    logging.info(f"CSV лог инициализирован: {csv_filename}")
    return current_backup, log_filename, csv_filename

# ------------- Функции формирования батча -------------
def collate_fn(batch, max_length=None):
    waveforms = []
    sample_rates = []
    transcripts = []  # оставляем для совместимости
    for item in batch:
        waveform, sample_rate, transcript, *_ = item
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val
        if max_length is not None and waveform.size(-1) > max_length:
            waveform = waveform[..., :max_length]
        waveforms.append(waveform)
        sample_rates.append(sample_rate)
        transcripts.append(transcript)
    batch_max_length = max(w.size(-1) for w in waveforms)
    padded_waveforms = [F.pad(w, (0, batch_max_length - w.size(-1))) for w in waveforms]
    waveforms = torch.stack(padded_waveforms, dim=0)
    return waveforms, sample_rates[0], transcripts

def collate_fn_max(batch):
    MAX_LENGTH = 4 * 16000  # 4 секунды при 16 кГц
    return collate_fn(batch, max_length=MAX_LENGTH)

# ------------- Определение модели – автоэнкодер для голосового слепка -------------
class VoiceAutoencoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 hidden_channels=128,
                 latent_dim=64,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 use_batch_norm=False,
                 dropout_rate=0.0):
        """
        Параметры:
          in_channels        : число аудиоканалов (обычно 1)
          hidden_channels    : число фильтров во внутренних слоях
          latent_dim         : размер латентного представления (голосовой слепок)
          num_encoder_layers : число слоёв в энкодере
          num_decoder_layers : число слоёв в декодере
          use_batch_norm     : если True – используется BatchNorm
          dropout_rate       : если > 0 – применяется Dropout
        """
        super(VoiceAutoencoder, self).__init__()
        
        # --- Энкодер ---
        encoder_layers = []
        if num_encoder_layers == 1:
            encoder_layers.append(nn.Conv1d(in_channels, latent_dim,
                                            kernel_size=4, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())
        else:
            encoder_layers.append(nn.Conv1d(in_channels, hidden_channels,
                                            kernel_size=4, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_channels))
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            for _ in range(num_encoder_layers - 2):
                encoder_layers.append(nn.Conv1d(hidden_channels, hidden_channels,
                                                kernel_size=4, stride=2, padding=1))
                encoder_layers.append(nn.ReLU())
                if use_batch_norm:
                    encoder_layers.append(nn.BatchNorm1d(hidden_channels))
                if dropout_rate > 0:
                    encoder_layers.append(nn.Dropout(dropout_rate))
            encoder_layers.append(nn.Conv1d(hidden_channels, latent_dim,
                                            kernel_size=4, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(latent_dim))
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # --- Декодер ---
        decoder_layers = []
        if num_decoder_layers == 1:
            decoder_layers.append(nn.ConvTranspose1d(latent_dim, in_channels,
                                                     kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.Tanh())
        else:
            decoder_layers.append(nn.ConvTranspose1d(latent_dim, hidden_channels,
                                                     kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.ReLU())
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_channels))
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            for _ in range(num_decoder_layers - 2):
                decoder_layers.append(nn.ConvTranspose1d(hidden_channels, hidden_channels,
                                                         kernel_size=4, stride=2, padding=1))
                decoder_layers.append(nn.ReLU())
                if use_batch_norm:
                    decoder_layers.append(nn.BatchNorm1d(hidden_channels))
                if dropout_rate > 0:
                    decoder_layers.append(nn.Dropout(dropout_rate))
            decoder_layers.append(nn.ConvTranspose1d(hidden_channels, in_channels,
                                                     kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def get_voice_imprint(self, x):
        """
        Извлекает голосовой слепок:
          - пропускает сигнал через энкодер,
          - усредняет по временной оси,
          - возвращает тензор (B, latent_dim)
        """
        latent = self.encoder(x)            # (B, latent_dim, T')
        imprint = torch.mean(latent, dim=2)   # (B, latent_dim)
        return imprint

# ------------- Функция обучения автоэнкодера -------------
def train_autoencoder(model, dataloader, optimizer, device, num_epochs, grad_clip, csv_filename):
    model.train()
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    prev_epoch_loss = None
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["epoch", "avg_loss", "delta", "learning_rate"])
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for waveforms, sr, transcripts in pbar:
                waveforms = waveforms.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(waveforms)
                    loss = criterion(outputs, waveforms)
                loss_value = loss.item()
                if math.isnan(loss_value):
                    logging.error("Loss стал NaN! Прерывание цикла обучения.")
                    return float('inf')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss_value
                pbar.set_postfix(loss=f"{loss_value:.6f}")
            avg_loss = epoch_loss / len(dataloader)
            current_lr = optimizer.param_groups[0]["lr"]
            delta = (prev_epoch_loss - avg_loss) if prev_epoch_loss is not None else 0.0
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.6f}, Delta: {delta:.6f}, LR: {current_lr}")
            csv_writer.writerow([epoch+1, avg_loss, delta, current_lr])
            prev_epoch_loss = avg_loss
    return avg_loss

# ------------- Функция извлечения и сохранения голосовых слепков -------------
def extract_and_save_imprints(model, dataloader, device, output_dir="imprints"):
    output_dir = os.path.join(BASE_DIR, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    index = 0
    model.eval()
    with torch.no_grad():
        for waveforms, sr, transcripts in tqdm(dataloader, desc="Извлечение слепков"):
            waveforms = waveforms.to(device)
            imprints = model.get_voice_imprint(waveforms)
            imprints_np = imprints.cpu().numpy()
            for i in range(imprints_np.shape[0]):
                file_path = os.path.join(output_dir, f"imprint_{index}.npz")
                np.savez_compressed(file_path, imprint=imprints_np[i])
                logging.info(f"Сохранён слепок: {file_path}")
                index += 1

# ------------- Функция для оценки индивидуальных гиперпараметров -------------
def evaluate_individual(hyperparams, trial_id=None):
    temp_backup = os.path.join(BASE_DIR, "optuna_trials" if trial_id is not None else "evo_trials",
                               f"trial_{trial_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(temp_backup, exist_ok=True)
    csv_filename = os.path.join(temp_backup, "training_metrics.csv")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    dataset = torchaudio.datasets.LIBRISPEECH(root=data_dir, url="train-clean-100", download=True)
    subset_indices = list(range(200))
    subset_dataset = Subset(dataset, subset_indices)
    dataloader = DataLoader(
        subset_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_max
    )
    
    model = VoiceAutoencoder(
        in_channels=hyperparams["in_channels"],
        hidden_channels=hyperparams["hidden_channels"],
        latent_dim=hyperparams["latent_dim"],
        num_encoder_layers=hyperparams["num_encoder_layers"],
        num_decoder_layers=hyperparams["num_decoder_layers"],
        use_batch_norm=hyperparams["use_batch_norm"],
        dropout_rate=hyperparams["dropout_rate"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    final_loss = train_autoencoder(model, dataloader, optimizer, device,
                                   num_epochs=hyperparams["num_epochs"],
                                   grad_clip=hyperparams["grad_clip"],
                                   csv_filename=csv_filename)
    if math.isinf(final_loss) or math.isnan(final_loss):
        return float('inf')
    return final_loss

# ------------- Режим одиночного обучения (train) -------------
def train_main(args):
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    backup_dir, log_filename, csv_filename = setup_logging("backup")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Используется устройство: {device}")
    
    training_params = {
        "in_channels": args.in_channels,
        "hidden_channels": args.hidden_channels,
        "latent_dim": args.latent_dim,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "use_batch_norm": args.use_batch_norm,
        "dropout_rate": args.dropout_rate,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "grad_clip": args.grad_clip,
    }
    logging.info(f"Параметры обучения: {training_params}")
    
    dataset = torchaudio.datasets.LIBRISPEECH(root=data_dir, url="train-clean-100", download=True)
    subset_indices = list(range(200))
    subset_dataset = Subset(dataset, subset_indices)
    dataloader = DataLoader(
        subset_dataset,
        batch_size=training_params["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_max
    )
    
    model = VoiceAutoencoder(
        in_channels=training_params["in_channels"],
        hidden_channels=training_params["hidden_channels"],
        latent_dim=training_params["latent_dim"],
        num_encoder_layers=training_params["num_encoder_layers"],
        num_decoder_layers=training_params["num_decoder_layers"],
        use_batch_norm=training_params["use_batch_norm"],
        dropout_rate=training_params["dropout_rate"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params["learning_rate"])
    logging.info("Начало обучения автоэнкодера...")
    final_loss = train_autoencoder(model, dataloader, optimizer, device,
                                   num_epochs=training_params["num_epochs"],
                                   grad_clip=training_params["grad_clip"],
                                   csv_filename=csv_filename)
    if math.isinf(final_loss):
        logging.error("Цикл обучения завершился с NaN loss. Завершаем работу.")
        return
    model_path = os.path.join(backup_dir, "voice_autoencoder.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Модель сохранена: {model_path}")
    
    imprints_dir = os.path.join(backup_dir, "imprints")
    extract_and_save_imprints(model, dataloader, device, output_dir=imprints_dir)

# ------------- Режим извлечения (extract) -------------
def extract_main(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Используется устройство: {device}")
    
    model = VoiceAutoencoder(
        in_channels=1,
        hidden_channels=128,
        latent_dim=64,
        num_encoder_layers=3,
        num_decoder_layers=3,
        use_batch_norm=False,
        dropout_rate=0.0
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f"Загружена модель из: {model_path}")
    
    data_dir = os.path.join(BASE_DIR, "data")
    dataset = torchaudio.datasets.LIBRISPEECH(root=data_dir, url="test-clean", download=True)
    subset_indices = list(range(50))
    subset_dataset = Subset(dataset, subset_indices)
    dataloader = DataLoader(
        subset_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_max
    )
    
    output_dir = os.path.join(BASE_DIR, "extracted_imprints")
    extract_and_save_imprints(model, dataloader, device, output_dir=output_dir)

# ------------- Режим бесконечного обучения (infinite) -------------
def infinite_train_main(args):
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Разбиваем параметры на группы
    train_params = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "grad_clip": args.grad_clip,
    }
    advanced_params = {
        "dropout_rate": args.dropout_rate,
    }
    arch_params = {
        "in_channels": args.in_channels,
        "hidden_channels": args.hidden_channels,
        "latent_dim": args.latent_dim,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "use_batch_norm": args.use_batch_norm,
    }
    
    best_loss = float('inf')
    cycle = 0
    while True:
        cycle += 1
        print(f"\n===== Цикл {cycle} =====")
        current_params = {}
        current_params.update(arch_params)
        current_params.update(train_params)
        current_params.update(advanced_params)
        print(f"Текущие параметры: {current_params}")
        
        backup_folder = os.path.join(BASE_DIR, "backup", f"cycle_{cycle}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(backup_folder, exist_ok=True)
        csv_filename = os.path.join(backup_folder, "training_metrics.csv")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        dataset = torchaudio.datasets.LIBRISPEECH(root=data_dir, url="train-clean-100", download=True)
        subset_indices = list(range(200))
        subset_dataset = Subset(dataset, subset_indices)
        dataloader = DataLoader(
            subset_dataset,
            batch_size=train_params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn_max
        )
        
        model = VoiceAutoencoder(
            in_channels=arch_params["in_channels"],
            hidden_channels=arch_params["hidden_channels"],
            latent_dim=arch_params["latent_dim"],
            num_encoder_layers=arch_params["num_encoder_layers"],
            num_decoder_layers=arch_params["num_decoder_layers"],
            use_batch_norm=arch_params["use_batch_norm"],
            dropout_rate=advanced_params["dropout_rate"]
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=train_params["learning_rate"])
        logging.info("Начало обучения автоэнкодера в бесконечном режиме...")
        final_loss = train_autoencoder(model, dataloader, optimizer, device,
                                       num_epochs=train_params["num_epochs"],
                                       grad_clip=train_params["grad_clip"],
                                       csv_filename=csv_filename)
        print(f"Цикл {cycle} завершён. Финальная loss: {final_loss:.6f}")
        
        if math.isinf(final_loss):
            print("Loss равен inf (NaN) – этот цикл не дал результата. Переход к следующему.")
        elif final_loss < best_loss:
            best_loss = final_loss
            best_model_path = os.path.join(backup_folder, "best_voice_autoencoder.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Новый лучший результат! Loss: {best_loss:.6f}. Модель сохранена в {best_model_path}")
        
        new_lr = train_params["learning_rate"] * random.uniform(0.8, 1.2)
        train_params["learning_rate"] = max(1e-5, min(new_lr, 1e-3))
        train_params["num_epochs"] = random.choice([10, 15, 20])
        new_dropout = advanced_params["dropout_rate"] + random.uniform(-0.05, 0.05)
        advanced_params["dropout_rate"] = max(0.0, min(new_dropout, 0.5))
        
        print("Новые параметры для следующего цикла:")
        print(f"  learning_rate: {train_params['learning_rate']:.6f}")
        print(f"  num_epochs   : {train_params['num_epochs']}")
        print(f"  dropout_rate : {advanced_params['dropout_rate']:.2f}")

# ------------- Режим оптимизации через Optuna -------------
def optuna_train_main(args):
    if optuna is None:
        print("Optuna не установлен. Установите optuna и повторите попытку.")
        return

    def objective(trial):
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        batch_size = trial.suggest_int("batch_size", 8, 32)
        num_epochs = trial.suggest_int("num_epochs", 10, 20)
        grad_clip = trial.suggest_uniform("grad_clip", 0.5, 2.0)
        dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.5)
        hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
        latent_dim = trial.suggest_int("latent_dim", 32, 128)
        num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 4)
        num_decoder_layers = trial.suggest_int("num_decoder_layers", 2, 4)
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [False, True])
        
        hyperparams = {
            "in_channels": 1,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "grad_clip": grad_clip,
            "dropout_rate": dropout_rate,
            "hidden_channels": hidden_channels,
            "latent_dim": latent_dim,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "use_batch_norm": use_batch_norm,
        }
        loss = evaluate_individual(hyperparams, trial_id=trial.number)
        return loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)
    print("Лучший trial:")
    best_trial = study.best_trial
    print("  Loss:", best_trial.value)
    print("  Параметры:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

# ------------- Функции для эволюционной оптимизации -------------
def random_individual():
    return {
        "in_channels": 1,
        "learning_rate": math.exp(random.uniform(math.log(1e-5), math.log(1e-3))),
        "batch_size": random.choice([8, 10, 16, 32]),
        "num_epochs": random.choice([10, 15, 20]),
        "grad_clip": random.uniform(0.5, 2.0),
        "dropout_rate": random.uniform(0.0, 0.5),
        "hidden_channels": random.choice([64, 128, 256]),
        "latent_dim": random.randint(32, 128),
        "num_encoder_layers": random.randint(2, 4),
        "num_decoder_layers": random.randint(2, 4),
        "use_batch_norm": random.choice([False, True]),
    }

def crossover(ind1, ind2):
    child = {}
    for key in ind1:
        child[key] = random.choice([ind1[key], ind2[key]])
    return child

def mutate(ind):
    mutated = ind.copy()
    if random.random() < 0.5:
        mutated["learning_rate"] *= random.uniform(0.8, 1.2)
        mutated["learning_rate"] = max(1e-5, min(mutated["learning_rate"], 1e-3))
    if random.random() < 0.5:
        mutated["grad_clip"] *= random.uniform(0.8, 1.2)
        mutated["grad_clip"] = max(0.5, min(mutated["grad_clip"], 2.0))
    if random.random() < 0.5:
        mutated["dropout_rate"] += random.uniform(-0.05, 0.05)
        mutated["dropout_rate"] = max(0.0, min(mutated["dropout_rate"], 0.5))
    if random.random() < 0.5:
        mutated["latent_dim"] = random.randint(32, 128)
    if random.random() < 0.5:
        mutated["batch_size"] = random.choice([8, 10, 16, 32])
    if random.random() < 0.5:
        mutated["num_epochs"] = random.choice([10, 15, 20])
    if random.random() < 0.5:
        mutated["hidden_channels"] = random.choice([64, 128, 256])
    if random.random() < 0.5:
        mutated["num_encoder_layers"] = random.randint(2, 4)
    if random.random() < 0.5:
        mutated["num_decoder_layers"] = random.randint(2, 4)
    if random.random() < 0.2:
        mutated["use_batch_norm"] = not mutated["use_batch_norm"]
    return mutated

def evolutionary_train_main(args):
    pop_size = args.pop_size
    num_generations = args.num_generations
    population = [random_individual() for _ in range(pop_size)]
    best_individual = None
    best_loss = float('inf')
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    for generation in range(num_generations):
        print(f"\n===== Поколение {generation+1} =====")
        results = []
        for i, individual in enumerate(population):
            print(f"Оценка индивидуума {i+1}/{len(population)}: {individual}")
            loss = evaluate_individual(individual, trial_id=f"evo_gen{generation}_ind{i}")
            print(f"  Loss: {loss}")
            results.append((individual, loss))
        results.sort(key=lambda x: x[1])
        best_gen = results[0]
        if best_gen[1] < best_loss:
            best_loss = best_gen[1]
            best_individual = best_gen[0]
        print(f"Лучший в поколении: {best_gen[0]} с loss: {best_gen[1]}")
        survivors = [ind for ind, loss in results[:pop_size//2]]
        new_population = survivors.copy()
        while len(new_population) < pop_size:
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population
    print("Лучший индивид:", best_individual, "с loss:", best_loss)

# ------------- Основной блок с подкомандами -------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение и извлечение голосового слепка")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Выберите режим работы")
    
    # Подкоманда train
    train_parser = subparsers.add_parser("train", help="Одиночный цикл обучения")
    train_group = train_parser.add_argument_group("Training Parameters (non-critical)")
    train_group.add_argument("--learning_rate", type=float, default=1e-4, help="Скорость обучения")
    train_group.add_argument("--batch_size", type=int, default=10, help="Размер батча")
    train_group.add_argument("--num_epochs", type=int, default=15, help="Число эпох")
    train_group.add_argument("--grad_clip", type=float, default=1.0, help="Норма клиппинга градиентов")
    
    advanced_group = train_parser.add_argument_group("Advanced Parameters")
    advanced_group.add_argument("--dropout_rate", type=float, default=0.0, help="Скорость dropout")
    
    arch_group = train_parser.add_argument_group("Architecture Parameters (critical)")
    arch_group.add_argument("--in_channels", type=int, default=1, help="Число входных каналов")
    arch_group.add_argument("--hidden_channels", type=int, default=128, help="Число фильтров скрытых слоёв")
    arch_group.add_argument("--latent_dim", type=int, default=64, help="Размер латентного представления")
    arch_group.add_argument("--num_encoder_layers", type=int, default=3, help="Число слоёв энкодера")
    arch_group.add_argument("--num_decoder_layers", type=int, default=3, help="Число слоёв декодера")
    arch_group.add_argument("--use_batch_norm", action="store_true", help="Использовать BatchNorm")
    
    # Подкоманда extract
    extract_parser = subparsers.add_parser("extract", help="Извлечение голосовых слепков")
    extract_parser.add_argument("--model_path", type=str, required=True, help="Путь к сохранённой модели")
    
    # Подкоманда infinite
    infinite_parser = subparsers.add_parser("infinite", help="Бесконечное обучение с изменением гиперпараметров")
    inf_train_group = infinite_parser.add_argument_group("Training Parameters (non-critical)")
    inf_train_group.add_argument("--learning_rate", type=float, default=1e-4, help="Скорость обучения")
    inf_train_group.add_argument("--batch_size", type=int, default=10, help="Размер батча")
    inf_train_group.add_argument("--num_epochs", type=int, default=15, help="Число эпох")
    inf_train_group.add_argument("--grad_clip", type=float, default=1.0, help="Норма клиппинга градиентов")
    
    inf_advanced_group = infinite_parser.add_argument_group("Advanced Parameters")
    inf_advanced_group.add_argument("--dropout_rate", type=float, default=0.0, help="Скорость dropout")
    
    inf_arch_group = infinite_parser.add_argument_group("Architecture Parameters (critical)")
    inf_arch_group.add_argument("--in_channels", type=int, default=1, help="Число входных каналов")
    inf_arch_group.add_argument("--hidden_channels", type=int, default=128, help="Число фильтров скрытых слоёв")
    inf_arch_group.add_argument("--latent_dim", type=int, default=64, help="Размер латентного представления")
    inf_arch_group.add_argument("--num_encoder_layers", type=int, default=3, help="Число слоёв энкодера")
    inf_arch_group.add_argument("--num_decoder_layers", type=int, default=3, help="Число слоёв декодера")
    inf_arch_group.add_argument("--use_batch_norm", action="store_true", help="Использовать BatchNorm")
    
    # Подкоманда optuna
    optuna_parser = subparsers.add_parser("optuna", help="Оптимизация гиперпараметров с помощью Optuna")
    optuna_parser.add_argument("--n_trials", type=int, default=10, help="Число испытаний")
    
    # Подкоманда evolutionary
    evolutionary_parser = subparsers.add_parser("evolutionary", help="Эволюционная оптимизация гиперпараметров")
    evolutionary_parser.add_argument("--pop_size", type=int, default=4, help="Размер популяции")
    evolutionary_parser.add_argument("--num_generations", type=int, default=3, help="Число поколений")
    evolutionary_parser.add_argument("--learning_rate", type=float, default=1e-4, help="Скорость обучения")
    evolutionary_parser.add_argument("--batch_size", type=int, default=10, help="Размер батча")
    evolutionary_parser.add_argument("--num_epochs", type=int, default=15, help="Число эпох")
    evolutionary_parser.add_argument("--grad_clip", type=float, default=1.0, help="Норма клиппинга градиентов")
    evolutionary_parser.add_argument("--dropout_rate", type=float, default=0.0, help="Скорость dropout")
    evolutionary_parser.add_argument("--in_channels", type=int, default=1, help="Число входных каналов")
    evolutionary_parser.add_argument("--hidden_channels", type=int, default=128, help="Число фильтров скрытых слоёв")
    evolutionary_parser.add_argument("--latent_dim", type=int, default=64, help="Размер латентного представления")
    evolutionary_parser.add_argument("--num_encoder_layers", type=int, default=3, help="Число слоёв энкодера")
    evolutionary_parser.add_argument("--num_decoder_layers", type=int, default=3, help="Число слоёв декодера")
    evolutionary_parser.add_argument("--use_batch_norm", action="store_true", help="Использовать BatchNorm")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_main(args)
    elif args.mode == "extract":
        extract_main(args.model_path)
    elif args.mode == "infinite":
        infinite_train_main(args)
    elif args.mode == "optuna":
        optuna_train_main(args)
    elif args.mode == "evolutionary":
        evolutionary_train_main(args)
