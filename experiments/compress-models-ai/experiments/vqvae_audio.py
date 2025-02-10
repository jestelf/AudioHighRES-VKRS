#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пример обучения сверточного автоэнкодера для реконструкции аудио на подмножестве датасета LibriSpeech с:
  - Информативным выводом прогресса (tqdm)
  - Смешанным обучением (mixed precision training с torch.amp)
  - Проверкой использования GPU
  - Параметризацией числа слоёв и внутренних параметров слоёв с подробными комментариями

Изменения:
  - Используется LibriSpeech "train-clean-100", но для демонстрации обучается только на подмножестве (например, первые 200 примеров)
  - Нормализация аудио (деление на максимум) и усечение сигналов до 4 секунд (при 16 кГц)
  - Выходной слой декодера с Tanh ограничивает амплитуду в диапазоне [-1, 1] для предотвращения перепиков
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm  # Прогресс-бар

# ------------- Функция collate_fn и её обёртка -------------
def collate_fn(batch, max_length=None):
    """
    Каждый элемент batch из LibriSpeech имеет вид:
       (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
    Действия:
      - Нормализует аудио (делит на абсолютный максимум, чтобы диапазон стал [-1, 1])
      - Усечение сигнала до max_length (если задано)
      - Padding для выравнивания длин сигналов в батче
    """
    waveforms = []
    sample_rates = []
    transcripts = []
    for item in batch:
        waveform, sample_rate, transcript, *_ = item
        # Нормализация: делим на абсолютное значение максимума, чтобы сигнал оказался в диапазоне [-1, 1]
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val
        # Усечение сигнала, если он длиннее max_length
        if max_length is not None and waveform.size(-1) > max_length:
            waveform = waveform[..., :max_length]
        waveforms.append(waveform)
        sample_rates.append(sample_rate)
        transcripts.append(transcript)
    batch_max_length = max(w.size(-1) for w in waveforms)
    padded_waveforms = []
    for w in waveforms:
        if w.size(-1) < batch_max_length:
            w = F.pad(w, (0, batch_max_length - w.size(-1)))
        padded_waveforms.append(w)
    waveforms = torch.stack(padded_waveforms, dim=0)
    return waveforms, sample_rates[0], transcripts

def collate_fn_max(batch):
    MAX_LENGTH = 4 * 16000  # Ограничение длины аудио: 4 секунды при 16 кГц
    return collate_fn(batch, max_length=MAX_LENGTH)

# ------------- Определение автоэнкодера с параметризацией слоёв -------------
class AudioAutoencoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=128, latent_dim=64,
                 num_encoder_layers=3, num_decoder_layers=3,
                 encoder_kernel_sizes=None, encoder_strides=None, encoder_paddings=None,
                 decoder_kernel_sizes=None, decoder_strides=None, decoder_paddings=None,
                 use_batch_norm=False, dropout_rate=0.0):
        """
        in_channels: число входных аудиоканалов (1 для моно, 2 для стерео).
        hidden_channels: число фильтров для промежуточных слоёв.
        latent_dim: размер латентного представления, получаемого в конце энкодера.
        
        num_encoder_layers: число слоёв в энкодере.
          Если равно 3 (по умолчанию), то:
            - Первый слой: преобразует in_channels -> hidden_channels.
            - Промежуточные слои (если num_encoder_layers > 2): сохраняют число каналов = hidden_channels.
            - Последний слой: преобразует hidden_channels -> latent_dim.
        num_decoder_layers: число слоёв в декодере.
          Если равно 3 (по умолчанию), то:
            - Первый слой: преобразует latent_dim -> hidden_channels.
            - Промежуточные слои: сохраняют число каналов = hidden_channels.
            - Последний слой: преобразует hidden_channels -> in_channels и применяет Tanh.
        
        encoder_kernel_sizes, encoder_strides, encoder_paddings:
          Либо списки длины num_encoder_layers, либо одно значение, которое будет использоваться для всех слоёв.
          По умолчанию: kernel_size=4, stride=2, padding=1.
          
        decoder_kernel_sizes, decoder_strides, decoder_paddings:
          Аналогично для декодера (ConvTranspose1d).
        
        use_batch_norm: если True, после каждого свёрточного слоя добавляется BatchNorm1d для улучшения стабильности обучения.
        dropout_rate: если > 0, после каждого слоя добавляется Dropout (значение от 0 до 1) для регуляризации.
        """
        super(AudioAutoencoder, self).__init__()
        
        # Если параметры для слоёв не заданы, используем дефолтные значения
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
        # Если только один слой в энкодере: сразу from in_channels -> latent_dim
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
            # Первый слой: from in_channels -> hidden_channels
            encoder_layers.append(nn.Conv1d(in_channels, hidden_channels,
                                            kernel_size=encoder_kernel_sizes[0],
                                            stride=encoder_strides[0],
                                            padding=encoder_paddings[0]))
            encoder_layers.append(nn.ReLU())
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_channels))
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            # Промежуточные слои (если num_encoder_layers > 2)
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
            # Последний слой: from hidden_channels -> latent_dim
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
            # Первый слой: from latent_dim -> hidden_channels
            decoder_layers.append(nn.ConvTranspose1d(latent_dim, hidden_channels,
                                                     kernel_size=decoder_kernel_sizes[0],
                                                     stride=decoder_strides[0],
                                                     padding=decoder_paddings[0]))
            decoder_layers.append(nn.ReLU())
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_channels))
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            # Промежуточные слои
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
            # Последний слой: from hidden_channels -> in_channels, затем Tanh для ограничения амплитуды
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

# ------------- Функция обучения с mixed precision и tqdm -------------
def train_autoencoder(model, dataloader, optimizer, device, num_epochs=15):
    """
    num_epochs: число эпох обучения. Чем больше эпох, тем точнее может сходиться модель, но обучение длится дольше.
    Mixed precision (torch.amp) используется для ускорения работы на GPU и экономии памяти.
    Дополнительно выводится дельта (разница) между потерями текущей и предыдущей эпох для анализа прогресса.
    """
    model.train()
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    
    prev_epoch_loss = None
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
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        avg_loss = epoch_loss / len(dataloader)
        if prev_epoch_loss is not None:
            delta = prev_epoch_loss - avg_loss
            if delta > 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f} (Improved by {delta:.6f})")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f} (Worsened by {-delta:.6f})")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")
        prev_epoch_loss = avg_loss

# ------------- Функция тестирования: сохраняем оригинальное и реконструированное аудио -------------
def test_autoencoder(model, dataloader, device, output_dir="output"):
    """
    Сохраняет первые 6 батчей: для каждого батча сохраняется первый пример (оригинал и реконструкция)
    для наглядного сравнения.
    """
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
            print(f"Сохранены: {orig_path} и {recon_path}")
            if idx >= 5:
                break

# ------------- Основная функция -------------
def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Проверка использования GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Используется GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("GPU не доступен, используется CPU")
    
    # Гиперпараметры:
    in_channels = 1              # 1 для моно; для стерео использовать 2.
    hidden_channels = 640        # Количество фильтров в скрытых слоях; влияет на способность захватывать детали.
    latent_dim = 64              # Размер латентного пространства; влияет на степень сжатия и уровень шума.
    
    # Новые параметры для управления количеством и внутренней настройкой слоёв:
    num_encoder_layers = 4       # Число слоёв в энкодере (по умолчанию 3).
    num_decoder_layers = 4       # Число слоёв в декодере (по умолчанию 3).
    # Если не передать списки, для всех слоёв будут использованы следующие значения:
    # Для энкодера: kernel_size=4, stride=2, padding=1.
    encoder_kernel_sizes = None  # Можно передать, например, [4, 4, 4]
    encoder_strides = None       # Например, [2, 2, 2]
    encoder_paddings = None      # Например, [1, 1, 1]
    # Для декодера: аналогично
    decoder_kernel_sizes = None
    decoder_strides = None
    decoder_paddings = None
    #
    use_batch_norm = False       # Если True, после каждого свёрточного слоя будет добавлен BatchNorm1d.
    dropout_rate = 0.0           # Если > 0, после каждого слоя будет добавлен Dropout для регуляризации.
    
    learning_rate = 1e-4         # Скорость обучения; влияет на скорость сходимости и стабильность.
    batch_size = 10               # Размер батча; влияет на стабильность градиентов и использование памяти.
    num_epochs = 20              # Количество эпох; больше эпох – лучшая сходимость, но дольше обучение.
    # max_length используется в collate_fn_max (4 секунды при 16 кГц)
    
    # Загружаем датасет LibriSpeech "train-clean-100" и используем подмножество для ускорения экспериментов.
    dataset = torchaudio.datasets.LIBRISPEECH(root="data", url="train-clean-100", download=True)
    subset_indices = list(range(200))  # Можно уменьшить (например, до 50) для ускорения отладки.
    subset_dataset = Subset(dataset, subset_indices)
    
    dataloader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,      # Подберите значение для вашей системы (на Windows часто 0 или 1)
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_fn_max  # Глобальная функция для корректной сериализации
    )
    
    # Создаём модель с параметризуемым числом слоёв и дополнительными опциями:
    model = AudioAutoencoder(
        in_channels, hidden_channels, latent_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        encoder_kernel_sizes=encoder_kernel_sizes,
        encoder_strides=encoder_strides,
        encoder_paddings=encoder_paddings,
        decoder_kernel_sizes=decoder_kernel_sizes,
        decoder_strides=decoder_strides,
        decoder_paddings=decoder_paddings,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Начало обучения автоэнкодера на подмножестве LibriSpeech...")
    train_autoencoder(model, dataloader, optimizer, device, num_epochs)
    
    print("Тестирование модели и сохранение примеров в папку output...")
    test_autoencoder(model, dataloader, device, output_dir="output")

if __name__ == "__main__":
    main()
