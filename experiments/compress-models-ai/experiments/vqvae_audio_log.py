#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пример обучения сверточного автоэнкодера для реконструкции аудио на подмножестве датасета LibriSpeech с:
  - Информативным выводом прогресса (tqdm)
  - Смешанным обучением (mixed precision training с torch.amp)
  - Проверкой использования GPU
  - Параметризацией числа слоёв и внутренних параметров слоёв с подробными комментариями
  - Подробным логированием (консоль, training.log, CSV-файл для построения графиков)
  - Резервное копирование логов и аудиопримеров в папку backup/<timestamp>
  - Категоризацией результатов: лучшие (good), худшие (bad) и средние (neutral) – по финальному average loss,
    копирование аудиофайлов в соответствующие подпапки с датой (YYYY-MM-DD).
  
Изменения:
  - Используется LibriSpeech "train-clean-100", обучается на подмножестве (например, первые 200 примеров)
  - Нормализация аудио (деление на максимум) и усечение сигналов до 4 секунд (при 16 кГц)
  - Выходной слой декодера с Tanh ограничивает амплитуду в диапазоне [-1, 1]
  - Логируются метрики обучения: номер эпохи, средний loss, дельта, текущий learning rate
  - Метрики сохраняются в CSV-файл для последующего анализа
"""

import os
import csv
import logging
import datetime
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ------------- Настройка логирования и резервного копирования -------------
def setup_logging(base_dir="backup"):
    # Формируем метку времени для создания уникальной папки
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_dir = os.path.join(base_dir, timestamp)
    os.makedirs(backup_dir, exist_ok=True)
    
    log_filename = os.path.join(backup_dir, "training.log")
    csv_filename = os.path.join(backup_dir, "training_metrics.csv")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info("Логирование инициализировано.")
    logging.info(f"Backup folder: {backup_dir}")
    
    # Инициализируем CSV-файл с заголовками для метрик
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["epoch", "avg_loss", "delta", "learning_rate"])
    logging.info(f"CSV лог инициализирован: {csv_filename}")
    return backup_dir, log_filename, csv_filename

# ------------- Функция collate_fn и её обёртка -------------
def collate_fn(batch, max_length=None):
    waveforms = []
    sample_rates = []
    transcripts = []
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
    padded_waveforms = [F.pad(w, (0, batch_max_length - w.size(-1))) if w.size(-1) < batch_max_length else w for w in waveforms]
    waveforms = torch.stack(padded_waveforms, dim=0)
    return waveforms, sample_rates[0], transcripts

def collate_fn_max(batch):
    MAX_LENGTH = 4 * 16000  # 4 секунды при 16 кГц
    return collate_fn(batch, max_length=MAX_LENGTH)

# ------------- Определение автоэнкодера с параметризацией слоёв -------------
class AudioAutoencoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=128, latent_dim=64,
                 num_encoder_layers=3, num_decoder_layers=3,
                 encoder_kernel_sizes=None, encoder_strides=None, encoder_paddings=None,
                 decoder_kernel_sizes=None, decoder_strides=None, decoder_paddings=None,
                 use_batch_norm=False, dropout_rate=0.0):
        """
        Параметры:
          in_channels: число входных аудиоканалов.
          hidden_channels: число фильтров для промежуточных слоёв.
          latent_dim: размер латентного представления.
          num_encoder_layers: число слоёв в энкодере.
          num_decoder_layers: число слоёв в декодере.
          encoder_kernel_sizes, encoder_strides, encoder_paddings: параметры свёрточных слоёв энкодера.
          decoder_kernel_sizes, decoder_strides, decoder_paddings: параметры слоёв декодера.
          use_batch_norm: если True, добавляется BatchNorm1d после свёрточных слоёв.
          dropout_rate: если > 0, добавляется Dropout.
        """
        super(AudioAutoencoder, self).__init__()
        
        if encoder_kernel_sizes is None:
            encoder_kernel_sizes = [4] * num_encoder_layers
        if encoder_strides is None:
            encoder_strides = [2] * num_encoder_layers
        if encoder_paddings is None:
            encoder_paddings = [1] * num_encoder_layers
        if decoder_kernel_sizes is None:
            decoder_kernel_sizes = [4] * num_decoder_layers
        if decoder_strides is None:
            decoder_strides = [2] * num_decoder_layers
        if decoder_paddings is None:
            decoder_paddings = [1] * num_decoder_layers
        
        # Построение энкодера
        encoder_layers = []
        if num_encoder_layers == 1:
            encoder_layers.append(nn.Conv1d(in_channels, latent_dim,
                                            kernel_size=encoder_kernel_sizes[0],
                                            stride=encoder_strides[0],
                                            padding=encoder_paddings[0]))
            encoder_layers.append(nn.ReLU())
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(latent_dim))
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
        else:
            encoder_layers.append(nn.Conv1d(in_channels, hidden_channels,
                                            kernel_size=encoder_kernel_sizes[0],
                                            stride=encoder_strides[0],
                                            padding=encoder_paddings[0]))
            encoder_layers.append(nn.ReLU())
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_channels))
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            for i in range(1, num_encoder_layers - 1):
                encoder_layers.append(nn.Conv1d(hidden_channels, hidden_channels,
                                                kernel_size=encoder_kernel_sizes[i],
                                                stride=encoder_strides[i],
                                                padding=encoder_paddings[i]))
                encoder_layers.append(nn.ReLU())
                if use_batch_norm:
                    encoder_layers.append(nn.BatchNorm1d(hidden_channels))
                if dropout_rate > 0:
                    encoder_layers.append(nn.Dropout(dropout_rate))
            encoder_layers.append(nn.Conv1d(hidden_channels, latent_dim,
                                            kernel_size=encoder_kernel_sizes[-1],
                                            stride=encoder_strides[-1],
                                            padding=encoder_paddings[-1]))
            encoder_layers.append(nn.ReLU())
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(latent_dim))
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Построение декодера
        decoder_layers = []
        if num_decoder_layers == 1:
            decoder_layers.append(nn.ConvTranspose1d(latent_dim, in_channels,
                                                     kernel_size=decoder_kernel_sizes[0],
                                                     stride=decoder_strides[0],
                                                     padding=decoder_paddings[0]))
            decoder_layers.append(nn.Tanh())
        else:
            decoder_layers.append(nn.ConvTranspose1d(latent_dim, hidden_channels,
                                                     kernel_size=decoder_kernel_sizes[0],
                                                     stride=decoder_strides[0],
                                                     padding=decoder_paddings[0]))
            decoder_layers.append(nn.ReLU())
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_channels))
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            for i in range(1, num_decoder_layers - 1):
                decoder_layers.append(nn.ConvTranspose1d(hidden_channels, hidden_channels,
                                                         kernel_size=decoder_kernel_sizes[i],
                                                         stride=decoder_strides[i],
                                                         padding=decoder_paddings[i]))
                decoder_layers.append(nn.ReLU())
                if use_batch_norm:
                    decoder_layers.append(nn.BatchNorm1d(hidden_channels))
                if dropout_rate > 0:
                    decoder_layers.append(nn.Dropout(dropout_rate))
            decoder_layers.append(nn.ConvTranspose1d(hidden_channels, in_channels,
                                                     kernel_size=decoder_kernel_sizes[-1],
                                                     stride=decoder_strides[-1],
                                                     padding=decoder_paddings[-1]))
            decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# ------------- Функция обучения с mixed precision, gradient clipping и подробным логированием -------------
def train_autoencoder(model, dataloader, optimizer, device, num_epochs=15, grad_clip=1.0, csv_filename="training_metrics.csv"):
    """
    num_epochs: число эпох обучения.
    grad_clip: максимальная норма градиентов для клиппинга.
    Логирование: вывод дельты (разницы) между потерями текущей и предыдущей эпох, текущий learning rate и т.д.
    Метрики записываются в CSV-файл для последующего анализа.
    """
    model.train()
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    
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
                with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    outputs = model(waveforms)
                    loss = criterion(outputs, waveforms)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.6f}")
            avg_loss = epoch_loss / len(dataloader)
            current_lr = optimizer.param_groups[0]["lr"]
            if prev_epoch_loss is not None:
                delta = prev_epoch_loss - avg_loss
                if delta > 0:
                    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f} (Improved by {delta:.6f})")
                else:
                    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f} (Worsened by {-delta:.6f})")
            else:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")
            csv_writer.writerow([epoch+1, avg_loss, (prev_epoch_loss - avg_loss) if prev_epoch_loss is not None else 0.0, current_lr])
            prev_epoch_loss = avg_loss

# ------------- Функция тестирования: сохраняем примеры аудио -------------
def test_autoencoder(model, dataloader, device, output_dir="output"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (waveforms, sr, transcripts) in enumerate(dataloader):
            waveforms = waveforms.to(device)
            outputs = model(waveforms)
            original = waveforms[0].cpu()
            reconstructed = outputs[0].cpu()
            orig_path = os.path.join(output_dir, f"original_{idx}.wav")
            recon_path = os.path.join(output_dir, f"reconstructed_{idx}.wav")
            torchaudio.save(orig_path, original, sr)
            torchaudio.save(recon_path, reconstructed, sr)
            logging.info(f"Saved: {orig_path} and {recon_path}")
            if idx >= 5:
                break

# ------------- Функция для категоризации результатов -------------
def categorize_results(csv_filename, output_dir, backup_dir):
    """
    Из CSV-файла выбирается финальное значение average loss.
    В зависимости от пороговых значений (good_threshold, bad_threshold) копируются все аудиофайлы из output_dir в:
      - папку good, если final loss ниже порога,
      - папку bad, если выше,
      - или neutral, если между.
    В backup_dir создается подпапка с датой (YYYY-MM-DD) и соответствующей категорией.
    """
    good_threshold = 0.0070
    bad_threshold = 0.0080

    with open(csv_filename, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        if not reader:
            logging.info("CSV файл пуст!")
            return
        final_row = reader[-1]
        final_loss = float(final_row["avg_loss"])
    
    if final_loss < good_threshold:
        category = "good"
    elif final_loss > bad_threshold:
        category = "bad"
    else:
        category = "neutral"
    
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    dest_dir = os.path.join(backup_dir, date_str, category)
    os.makedirs(dest_dir, exist_ok=True)
    
    for filename in os.listdir(output_dir):
        if filename.endswith(".wav"):
            shutil.copy(os.path.join(output_dir, filename), dest_dir)
    # Заменили символ "→" на "->" для избежания проблем с кодировкой
    logging.info(f"Final loss = {final_loss:.6f} -> Category: '{category}'. Files copied to {dest_dir}")

# ------------- Основная функция -------------
def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    backup_dir, log_filename, csv_filename = setup_logging()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("GPU not available, using CPU")
    
    # Логируем параметры обучения
    training_params = {
        "in_channels": 1,
        "hidden_channels": 256,
        "latent_dim": 64,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "use_batch_norm": False,
        "dropout_rate": 0.0,
        "learning_rate": 1e-4,
        "batch_size": 10,
        "num_epochs": 40
    }
    logging.info(f"Training parameters: {training_params}")
    
    in_channels = training_params["in_channels"]
    hidden_channels = training_params["hidden_channels"]
    latent_dim = training_params["latent_dim"]
    num_encoder_layers = training_params["num_encoder_layers"]
    num_decoder_layers = training_params["num_decoder_layers"]
    use_batch_norm = training_params["use_batch_norm"]
    dropout_rate = training_params["dropout_rate"]
    
    learning_rate = training_params["learning_rate"]
    batch_size = training_params["batch_size"]
    num_epochs = training_params["num_epochs"]
    
    dataset = torchaudio.datasets.LIBRISPEECH(root="data", url="train-clean-100", download=True)
    subset_indices = list(range(200))
    subset_dataset = Subset(dataset, subset_indices)
    
    dataloader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_fn_max
    )
    
    model = AudioAutoencoder(
        in_channels, hidden_channels, latent_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        encoder_kernel_sizes=None,
        encoder_strides=None,
        encoder_paddings=None,
        decoder_kernel_sizes=None,
        decoder_strides=None,
        decoder_paddings=None,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    logging.info("Starting training on LibriSpeech subset...")
    train_autoencoder(model, dataloader, optimizer, device, num_epochs, grad_clip=1.0, csv_filename=csv_filename)
    
    logging.info("Testing model and saving examples to output folder...")
    test_autoencoder(model, dataloader, device, output_dir="output")
    
    categorize_results(csv_filename, output_dir="output", backup_dir=backup_dir)
    
if __name__ == "__main__":
    main()
