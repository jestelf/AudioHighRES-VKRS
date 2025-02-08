#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пример обучения сверточного автоэнкодера для реконструкции аудио на подмножестве датасета LibriSpeech.

Изменения:
  - Используется датасет LibriSpeech "train-clean-100", но для демонстрации обучения берется только подмножество (например, первые 200 примеров).
  - Нормализация аудио (каждый сигнал делится на свой максимум).
  - Ограничение длины аудио до 4 секунд (при 16 кГц) для удобства.
  - Выходной слой декодера с Tanh, чтобы амплитуда была в диапазоне [-1, 1].
  - Сохранение оригинала и реконструкции для сравнения.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Subset

# 1. Определение автоэнкодера
class AudioAutoencoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=128, latent_dim=64):
        super(AudioAutoencoder, self).__init__()
        # Encoder: 3 слоя Conv1d с downsampling (уменьшение длины примерно в 8 раз)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder: 3 слоя ConvTranspose1d для восстановления исходной размерности
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Ограничение амплитуды [-1, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# 2. Collate-функция для LibriSpeech
def collate_fn(batch, max_length=None):
    """
    Каждый элемент batch из LibriSpeech имеет вид:
       (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
    Функция:
      - Нормализует аудио (делит на абсолютный максимум, если не 0).
      - Усекает сигнал до max_length (если задано).
      - Дополняет (pad) аудиосигналы до одинаковой длины в батче.
    """
    waveforms = []
    sample_rates = []
    transcripts = []
    for item in batch:
        waveform, sample_rate, transcript, *_ = item
        # Нормализация (если максимум не 0)
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val
        # Усечение, если сигнал длиннее max_length
        if max_length is not None and waveform.size(-1) > max_length:
            waveform = waveform[..., :max_length]
        waveforms.append(waveform)
        sample_rates.append(sample_rate)
        transcripts.append(transcript)
    # Находим максимальную длину в батче
    batch_max_length = max(w.size(-1) for w in waveforms)
    padded_waveforms = []
    for w in waveforms:
        if w.size(-1) < batch_max_length:
            w = F.pad(w, (0, batch_max_length - w.size(-1)))
        padded_waveforms.append(w)
    waveforms = torch.stack(padded_waveforms, dim=0)
    # Предполагаем, что sample_rate у всех одинаковый
    return waveforms, sample_rates[0], transcripts

# 3. Функция обучения автоэнкодера
def train_autoencoder(model, dataloader, optimizer, device, num_epochs=5):
    model.train()
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for waveforms, sr, transcripts in dataloader:
            waveforms = waveforms.to(device)
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, waveforms)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

# 4. Функция тестирования: сохраняем оригинальное и реконструированное аудио для сравнения
def test_autoencoder(model, dataloader, device, output_dir="output"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (waveforms, sr, transcripts) in enumerate(dataloader):
            waveforms = waveforms.to(device)
            outputs = model(waveforms)
            # Сохраняем первый пример из батча
            original = waveforms[0].cpu()
            reconstructed = outputs[0].cpu()
            orig_path = os.path.join(output_dir, f"original_{idx}.wav")
            recon_path = os.path.join(output_dir, f"reconstructed_{idx}.wav")
            torchaudio.save(orig_path, original, sr)
            torchaudio.save(recon_path, reconstructed, sr)
            print(f"Сохранены: {orig_path} и {recon_path}")
            if idx >= 5:  # Сохраним несколько примеров
                break

# 5. Основная функция
def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Гиперпараметры
    in_channels = 1
    hidden_channels = 128
    latent_dim = 64
    learning_rate = 1e-3
    batch_size = 8
    num_epochs = 20
    # Ограничение длины аудио: 4 секунды при 16 кГц
    max_length = 4 * 16000
    
    # Загружаем датасет LibriSpeech "train-clean-100"
    dataset = torchaudio.datasets.LIBRISPEECH(root="data", url="train-clean-100", download=True)
    # Используем подмножество, чтобы тренировка шла быстрее (например, первые 200 примеров)
    subset_indices = list(range(200))
    subset_dataset = Subset(dataset, subset_indices)
    
    dataloader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, max_length)
    )
    
    model = AudioAutoencoder(in_channels, hidden_channels, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Начало обучения автоэнкодера на подмножестве LibriSpeech...")
    train_autoencoder(model, dataloader, optimizer, device, num_epochs)
    
    print("Тестирование модели и сохранение примеров в папку output...")
    test_autoencoder(model, dataloader, device, output_dir="output")

if __name__ == "__main__":
    main()
