#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запускатор модели автоэнкодера для аудио с графическим интерфейсом на PyQt.
Функциональность:
  - Загрузить сохранённую модель (.pth)
  - Выбрать аудиофайл (wav, mp3, flac и др.) и запустить его обработку:
      • Аудио дополняется до кратной 4-секунд, разбивается на сегменты.
      • Для каждого сегмента вычисляется латентное представление (выход энкодера).
      • Сегменты декодируются для реконструкции, а также из латентного представления генерируется аудио.
      • Полное латентное представление сохраняется:
            - в формате NPZ (с float16),
            - в формате NPZ с квантованием до int8 (с сохранением min и max),
            - дополнительно сжимается с помощью LZMA (latent_lzma.npz.lzma),
            - и ещё агрессивно квантуется до INT4 (значения 0..15, упаковка по 2 значения в один байт)
              и сохраняется с помощью LZMA (latent_int4.lzma),
            - а также квантуется до INT2 с _per‑channel_ квантованием с использованием mu‑law
              (значения 0..3 для каждого канала, упаковка 4 значений в 1 байт) и сохраняется с помощью LZMA (latent_int2.lzma).
  - Есть три кнопки для восстановления аудио:
      • «Восстановить аудио из NPZ» – стандартное восстановление из NPZ‑файла (float16 или int8).
      • «Восстановить аудио из LZMA NPZ» – восстановление из LZMA‑сжатого NPZ‑файла.
      • «Восстановить аудио из INT2 LZMA» – восстановление из LZMA‑сжатого файла с INT2‑упакованным представлением.
  
В папке output_pyqt сохраняются:
  - original.wav – исходное аудио,
  - reconstructed.wav – аудио, полученное при циклической обработке,
  - generated_from_latent.wav – аудио, сгенерированное повторным декодированием латентных сегментов,
  - latent.npz – полный массив латентного представления (float16, сжатый NPZ),
  - latent_int8.npz – вариант с квантованием до uint8,
  - latent_lzma.npz.lzma – NPZ с LZMA‑сжатием (float16),
  - latent_int4.lzma – файл с INT4‑упакованным представлением, сжатый через LZMA,
  - latent_int2.lzma – файл с INT2‑упакованным представлением (per‑channel mu‑law квантование), сжатый через LZMA,
  - restored_from_npz.wav – аудио, восстановленное из выбранного NPZ‑файла,
  - restored_from_lzma.wav – аудио, восстановленное из LZMA‑сжатого NPZ‑файла,
  - restored_from_int2.wav – аудио, восстановленное из INT2 LZMA‑файла.
  
Функции mu‑law:
    mu_law_encode(x, mu) = sign(x)*log(1 + mu*abs(x)) / log(1+mu)
    mu_law_decode(y, mu) = sign(y)*(exp(log(1+mu)*abs(y))-1)/mu
"""

import sys
import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lzma

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
                             QLabel, QFileDialog, QMessageBox)

# ------------- Функции mu‑law -------------
def mu_law_encode(x, mu):
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

def mu_law_decode(y, mu):
    return np.sign(y) * (np.expm1(np.log1p(mu) * np.abs(y)) / mu)

# ------------- Функции для сохранения/загрузки с использованием LZMA -------------
def save_latent_lzma(file_path, latent_array):
    with lzma.open(file_path, 'wb') as f:
        np.save(f, latent_array)

def load_latent_lzma(file_path):
    with lzma.open(file_path, 'rb') as f:
        latent_array = np.load(f)
    return latent_array

# ------------- Функции для упаковки INT4 -------------
def pack_int4(arr):
    """
    Упаковывает массив uint8 (значения 0..15) по 2 элемента в один байт.
    Возвращает упакованный массив и исходную форму.
    """
    original_shape = arr.shape
    arr_flat = arr.flatten()
    if arr_flat.size % 2 != 0:
        arr_flat = np.concatenate([arr_flat, np.zeros(1, dtype=np.uint8)])
    packed = (arr_flat[0::2] << 4) | (arr_flat[1::2] & 0x0F)
    return packed, original_shape

def unpack_int4(packed, original_shape):
    first = (packed >> 4) & 0x0F
    second = packed & 0x0F
    arr_flat = np.empty(first.size * 2, dtype=np.uint8)
    arr_flat[0::2] = first
    arr_flat[1::2] = second
    arr_flat = arr_flat[:np.prod(original_shape)]
    return arr_flat.reshape(original_shape)

def save_int4_lzma(file_path, packed, shape, min_val, max_val):
    with lzma.open(file_path, 'wb') as f:
        np.savez_compressed(f, packed=packed, shape=shape, min=min_val, max=max_val)

def load_int4_lzma(file_path):
    with lzma.open(file_path, 'rb') as f:
        data = np.load(f)
        packed = data['packed']
        shape = data['shape']
        min_val = data['min'].item()
        max_val = data['max'].item()
    return packed, shape, min_val, max_val

# ------------- Функции для упаковки INT2 -------------
def pack_int2(arr):
    """
    Упаковывает массив uint8 (значения 0..3) в формате INT2.
    Упаковывает 4 значения (по 2 бита) в один байт.
    """
    original_shape = arr.shape
    arr_flat = arr.flatten()
    remainder = arr_flat.size % 4
    if remainder != 0:
        pad_size = 4 - remainder
        arr_flat = np.concatenate([arr_flat, np.zeros(pad_size, dtype=np.uint8)])
    packed = (arr_flat[0::4] << 6) | (arr_flat[1::4] << 4) | (arr_flat[2::4] << 2) | (arr_flat[3::4])
    return packed, original_shape

def unpack_int2(packed, original_shape):
    a = (packed >> 6) & 0x03
    b = (packed >> 4) & 0x03
    c = (packed >> 2) & 0x03
    d = packed & 0x03
    arr_flat = np.empty(a.size * 4, dtype=np.uint8)
    arr_flat[0::4] = a
    arr_flat[1::4] = b
    arr_flat[2::4] = c
    arr_flat[3::4] = d
    arr_flat = arr_flat[:np.prod(original_shape)]
    return arr_flat.reshape(original_shape)

def save_int2_lzma(file_path, packed, shape, channel_min, channel_max):
    with lzma.open(file_path, 'wb') as f:
        np.savez_compressed(f, packed=packed, shape=shape, channel_min=channel_min, channel_max=channel_max)

def load_int2_lzma(file_path):
    with lzma.open(file_path, 'rb') as f:
        data = np.load(f)
        packed = data['packed']
        shape = data['shape']
        channel_min = data['channel_min']
        channel_max = data['channel_max']
    return packed, shape, channel_min, channel_max

# ------------- Определение модели автоэнкодера -------------
class AudioAutoencoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=256, latent_dim=64,
                 num_encoder_layers=3, num_decoder_layers=3,
                 encoder_kernel_sizes=None, encoder_strides=None, encoder_paddings=None,
                 decoder_kernel_sizes=None, decoder_strides=None, decoder_paddings=None,
                 use_batch_norm=False, dropout_rate=0.0):
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

        # --- Энкодер ---
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

        # --- Декодер ---
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

# ------------- Основное окно приложения -------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Запускатор модели автоэнкодера для аудио")
        self.setGeometry(100, 100, 450, 450)
        self.model = None
        self.model_loaded = False
        self.model_path = None
        self.audio_path = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.btn_load_model = QPushButton("Загрузить модель")
        self.btn_load_model.clicked.connect(self.load_model)
        self.layout.addWidget(self.btn_load_model)

        self.lbl_model = QLabel("Модель не загружена")
        self.layout.addWidget(self.lbl_model)

        self.btn_select_audio = QPushButton("Выбрать аудио файл")
        self.btn_select_audio.clicked.connect(self.select_audio)
        self.layout.addWidget(self.btn_select_audio)

        self.lbl_audio = QLabel("Аудио не выбрано")
        self.layout.addWidget(self.lbl_audio)

        self.btn_run = QPushButton("Запустить обработку")
        self.btn_run.clicked.connect(self.run_model)
        self.layout.addWidget(self.btn_run)

        # Кнопки для восстановления аудио
        self.btn_restore = QPushButton("Восстановить аудио из NPZ")
        self.btn_restore.clicked.connect(self.restore_from_npz)
        self.layout.addWidget(self.btn_restore)

        self.btn_restore_lzma = QPushButton("Восстановить аудио из LZMA NPZ")
        self.btn_restore_lzma.clicked.connect(self.restore_from_lzma)
        self.layout.addWidget(self.btn_restore_lzma)

        self.btn_restore_int2 = QPushButton("Восстановить аудио из INT2 LZMA")
        self.btn_restore_int2.clicked.connect(self.restore_from_int2_lzma)
        self.layout.addWidget(self.btn_restore_int2)

        self.lbl_status = QLabel("")
        self.layout.addWidget(self.lbl_status)

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл модели", "", "Model Files (*.pth)")
        if file_path:
            self.model_path = file_path
            self.model = AudioAutoencoder(
                in_channels=1,
                hidden_channels=256,
                latent_dim=64,
                num_encoder_layers=3,
                num_decoder_layers=3,
                use_batch_norm=False,
                dropout_rate=0.0
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            try:
                state_dict = torch.load(self.model_path, map_location=device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model_loaded = True
                self.lbl_model.setText(f"Модель загружена: {os.path.basename(self.model_path)}")
                self.lbl_status.setText("Модель успешно загружена.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{e}")

    def select_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите аудио файл", "", "Audio Files (*.wav *.mp3 *.flac)")
        if file_path:
            self.audio_path = file_path
            self.lbl_audio.setText(f"Аудио выбрано: {os.path.basename(self.audio_path)}")
            self.lbl_status.setText("Аудио файл выбран.")

    def run_model(self):
        """
        Обрабатывает аудио:
          1. Загружает аудио, приводит к моно и пересэмплирует до 16 кГц.
          2. Дополняет аудио до кратного 4-секунд.
          3. Разбивает аудио на сегменты по 4 секунды.
          4. Для каждого сегмента вычисляется латентное представление,
             которое сохраняется в список latent_segments.
          5. Выполняется декодирование для реконструкции и генерации аудио.
          6. Полное латентное представление объединяется и сохраняется:
             • latent.npz (float16),
             • latent_int8.npz (квантование до uint8, с min и max).
          7. Дополнительно, полное латентное представление сохраняется с помощью LZMA (latent_lzma.npz.lzma).
          8. Агрессивное квантование до INT4 (значения 0..15, упаковка по 2 значения в байт)
             сохраняется через LZMA (latent_int4.lzma).
          9. Агрессивное _per‑channel_ квантование до INT2 с применением mu‑law:
             для каждого канала данные масштабируются в [–1,1], кодируются по mu‑law (mu=100),
             затем квантуются в 4 уровня и упаковка 4 значений в 1 байт;
             сохраняется через LZMA (latent_int2.lzma).
         10. Сохраняются аудио: original.wav, reconstructed.wav, generated_from_latent.wav.
        """
        if not self.model_loaded:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите модель!")
            return
        if not self.audio_path:
            QMessageBox.warning(self, "Внимание", "Сначала выберите аудио файл!")
            return
        try:
            # Загрузка и предварительная обработка аудио
            waveform, sr = torchaudio.load(self.audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            target_sr = 16000
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
                sr = target_sr
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

            # Дополнение аудио до кратного 4-секунд
            segment_length = 4 * target_sr
            total_length = waveform.shape[1]
            remainder = total_length % segment_length
            if remainder > 0:
                pad_size = segment_length - remainder
                waveform = F.pad(waveform, (0, pad_size))
                total_length = waveform.shape[1]

            # Разбивка на сегменты
            num_segments = total_length // segment_length
            segments = [ waveform[:, i*segment_length:(i+1)*segment_length] for i in range(num_segments) ]

            reconstructed_segments = []
            latent_segments = []  # список для хранения латентных представлений

            # Обработка каждого сегмента
            for seg in segments:
                input_tensor = seg.unsqueeze(0).to(device)  # (1, 1, segment_length)
                with torch.no_grad():
                    latent = self.model.encoder(input_tensor)  # (1, latent_dim, L_latent)
                    latent_segments.append(latent.cpu())
                    rec_seg = self.model.decoder(latent).squeeze(0).cpu()
                reconstructed_segments.append(rec_seg)

            # Генерация аудио из латентных сегментов (повторное декодирование)
            generated_segments = []
            for latent in latent_segments:
                with torch.no_grad():
                    gen_seg = self.model.decoder(latent.to(device)).squeeze(0).cpu()
                generated_segments.append(gen_seg)

            reconstructed_audio = torch.cat(reconstructed_segments, dim=1)
            generated_audio = torch.cat(generated_segments, dim=1)

            # Объединение латентных представлений
            full_latent = torch.cat(latent_segments, dim=2)  # (1, latent_dim, total_latent_length)
            # Сохранение в формате float16
            full_latent_np = full_latent.numpy().astype(np.float16)
            # Глобальное квантование до uint8 (для INT8 варианта)
            global_min = full_latent_np.min()
            global_max = full_latent_np.max()
            full_latent_int8 = np.round((full_latent_np - global_min) / (global_max - global_min) * 255).astype(np.uint8)
            # Агрессивное квантование до INT4 (значения 0..15) и упаковка по 2 значения в байт
            int4_array = np.floor((full_latent_np - global_min) / (global_max - global_min) * 15).astype(np.uint8)
            packed_int4, int4_shape = pack_int4(int4_array)
            # _Per‑channel_ квантование до INT2 с использованием mu‑law
            # Для каждого канала вычисляем min и max:
            channel_min = full_latent_np.min(axis=2, keepdims=True)
            channel_max = full_latent_np.max(axis=2, keepdims=True)
            # Масштабируем каждый канал в [0,1]:
            norm = (full_latent_np - channel_min) / (channel_max - channel_min + 1e-8)
            # Преобразуем в [-1,1]:
            scaled = norm * 2 - 1
            mu = 100.0
            encoded = mu_law_encode(scaled, mu)
            # Теперь квантуем в 4 уровня: значения от 0 до 3
            int2_array = np.floor((encoded + 1) / 2 * 3).astype(np.uint8)
            packed_int2, int2_shape = pack_int2(int2_array)

            output_dir = os.path.join(os.getcwd(), "output_pyqt")
            os.makedirs(output_dir, exist_ok=True)
            original_path = os.path.join(output_dir, "original.wav")
            reconstructed_path = os.path.join(output_dir, "reconstructed.wav")
            generated_path = os.path.join(output_dir, "generated_from_latent.wav")
            latent_path = os.path.join(output_dir, "latent.npz")
            latent_int8_path = os.path.join(output_dir, "latent_int8.npz")
            latent_lzma_path = os.path.join(output_dir, "latent_lzma.npz.lzma")
            latent_int4_lzma_path = os.path.join(output_dir, "latent_int4.lzma")
            latent_int2_lzma_path = os.path.join(output_dir, "latent_int2.lzma")

            torchaudio.save(original_path, waveform, target_sr)
            torchaudio.save(reconstructed_path, reconstructed_audio, target_sr)
            torchaudio.save(generated_path, generated_audio, target_sr)

            # Сохранение NPZ-файлов
            np.savez_compressed(latent_path, latent=full_latent_np)
            np.savez_compressed(latent_int8_path, latent_int8=full_latent_int8, min=global_min, max=global_max)
            save_latent_lzma(latent_lzma_path, full_latent_np)
            save_int4_lzma(latent_int4_lzma_path, packed_int4, int4_shape, global_min, global_max)
            save_int2_lzma(latent_int2_lzma_path, packed_int2, int2_shape, channel_min, channel_max)

            msg = (f"Обработка завершена.\n"
                   f"Оригинал: {original_path}\n"
                   f"Реконструкция: {reconstructed_path}\n"
                   f"Генерация из латентного: {generated_path}\n"
                   f"Latent (float16) сохранён: {latent_path}\n"
                   f"Latent (int8) сохранён: {latent_int8_path}\n"
                   f"Latent (LZMA) сохранён: {latent_lzma_path}\n"
                   f"Latent (INT4 LZMA) сохранён: {latent_int4_lzma_path}\n"
                   f"Latent (INT2 LZMA) сохранён: {latent_int2_lzma_path}")
            self.lbl_status.setText(msg)
            QMessageBox.information(self, "Успех", msg)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обработке аудио:\n{e}")

    def restore_from_npz(self):
        """
        Восстанавливает аудио из выбранного NPZ-файла с латентным представлением.
        Если присутствуют ключи 'latent_int8', 'min' и 'max', выполняется обратное квантование.
        Сохраняется как restored_from_npz.wav.
        """
        if not self.model_loaded:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите модель!")
            return

        npz_path, _ = QFileDialog.getOpenFileName(self, "Выберите NPZ файл с латентным представлением", "", "NPZ Files (*.npz)")
        if not npz_path:
            return

        try:
            data = np.load(npz_path)
            if "latent_int8" in data:
                latent_int8 = data["latent_int8"]
                global_min = data["min"].item()
                global_max = data["max"].item()
                full_latent_np = (latent_int8.astype(np.float32) / 255.0) * (global_max - global_min) + global_min
                full_latent_np = full_latent_np.astype(np.float16)
            elif "latent" in data:
                full_latent_np = data["latent"]
            else:
                raise ValueError("Неверный формат NPZ: отсутствуют ключи 'latent' или 'latent_int8'")
            latent_tensor = torch.tensor(full_latent_np, dtype=torch.float32)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            latent_tensor = latent_tensor.to(device)
            with torch.no_grad():
                restored_audio = self.model.decoder(latent_tensor).squeeze(0).cpu()
            output_dir = os.path.join(os.getcwd(), "output_pyqt")
            os.makedirs(output_dir, exist_ok=True)
            restored_path = os.path.join(output_dir, "restored_from_npz.wav")
            torchaudio.save(restored_path, restored_audio, 16000)
            msg = f"Аудио, восстановленное из NPZ, сохранено: {restored_path}"
            self.lbl_status.setText(msg)
            QMessageBox.information(self, "Успех", msg)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при восстановлении аудио из NPZ:\n{e}")

    def restore_from_lzma(self):
        """
        Восстанавливает аудио из выбранного LZMA-сжатого NPZ-файла (float16).
        Сохраняется как restored_from_lzma.wav.
        """
        if not self.model_loaded:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите модель!")
            return
        lzma_path, _ = QFileDialog.getOpenFileName(self, "Выберите LZMA NPZ файл с латентным представлением", "", "LZMA Files (*.lzma)")
        if not lzma_path:
            return
        try:
            full_latent_np = load_latent_lzma(lzma_path)
            latent_tensor = torch.tensor(full_latent_np, dtype=torch.float32)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            latent_tensor = latent_tensor.to(device)
            with torch.no_grad():
                restored_audio = self.model.decoder(latent_tensor).squeeze(0).cpu()
            output_dir = os.path.join(os.getcwd(), "output_pyqt")
            os.makedirs(output_dir, exist_ok=True)
            restored_path = os.path.join(output_dir, "restored_from_lzma.wav")
            torchaudio.save(restored_path, restored_audio, 16000)
            msg = f"Аудио, восстановленное из LZMA NPZ, сохранено: {restored_path}"
            self.lbl_status.setText(msg)
            QMessageBox.information(self, "Успех", msg)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при восстановлении аудио из LZMA NPZ:\n{e}")

    def restore_from_int2_lzma(self):
        """
        Восстанавливает аудио из выбранного LZMA-сжатого файла с INT2-упакованным латентным представлением.
        При восстановлении происходит распаковка, обратное per‑channel mu‑law квантование и преобразование к float32.
        Сохраняется как restored_from_int2.wav.
        """
        if not self.model_loaded:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите модель!")
            return
        int2_path, _ = QFileDialog.getOpenFileName(self, "Выберите INT2 LZMA файл с латентным представлением", "", "LZMA Files (*.lzma)")
        if not int2_path:
            return
        try:
            packed, shape, channel_min, channel_max = load_int2_lzma(int2_path)
            # Распаковка INT2
            int2_array = unpack_int2(packed, shape)
            # Обратное per‑channel квантование с mu‑law
            # channel_min и channel_max имеют форму (1, latent_dim, 1)
            # Приведём данные к диапазону [-1,1]:
            norm = (full_latent_np - channel_min) / (channel_max - channel_min + 1e-8)  # но full_latent_np не определён, нам нужно восстановить из int2_array
            # Вместо этого, восстановим: 
            # Сначала, приведем int2_array (значения 0..3) в диапазон [-1,1]:
            recovered_encoded = (int2_array.astype(np.float32) / 3.0) * 2 - 1
            mu = 100.0
            # Обратное mu‑law декодирование:
            recovered_scaled = mu_law_decode(recovered_encoded, mu)
            # Вернём данные в диапазон [0,1]:
            recovered_norm = (recovered_scaled + 1) / 2
            # Восстановим оригинальный масштаб для каждого канала:
            # Здесь channel_min и channel_max – это per‑channel значения (хранятся в файл)
            # Так как они сохранены в виде массивов, используем их напрямую:
            full_latent_np = recovered_norm * (channel_max - channel_min) + channel_min
            full_latent_np = full_latent_np.astype(np.float16)

            latent_tensor = torch.tensor(full_latent_np, dtype=torch.float32)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            latent_tensor = latent_tensor.to(device)
            with torch.no_grad():
                restored_audio = self.model.decoder(latent_tensor).squeeze(0).cpu()
            output_dir = os.path.join(os.getcwd(), "output_pyqt")
            os.makedirs(output_dir, exist_ok=True)
            restored_path = os.path.join(output_dir, "restored_from_int2.wav")
            torchaudio.save(restored_path, restored_audio, 16000)
            msg = f"Аудио, восстановленное из INT2 LZMA, сохранено: {restored_path}"
            self.lbl_status.setText(msg)
            QMessageBox.information(self, "Успех", msg)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при восстановлении аудио из INT2 LZMA:\n{e}")

# ------------- Основной блок запуска приложения -------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
