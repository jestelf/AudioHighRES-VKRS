#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

# 1. Устанавливаем зависимости
for pkg in ["PyQt5", "TTS"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Устанавливаем {pkg} ...")
        install(pkg)

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
import numpy as np

import torch
import torch.nn as nn

from TTS.api import TTS

def main():
    # Инициализация Qt-приложения
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Выбор слепка голоса (*.npz)")
    window.resize(400, 300)
    window.show()

    # 2. Диалог выбора *.npz
    file_path, _ = QFileDialog.getOpenFileName(
        parent=window,
        caption="Выберите файл со слепком голоса (*.npz)",
        filter="NPZ Files (*.npz)"
    )
    if not file_path:
        print("Файл не выбран — завершаем.")
        return

    # 3. Считываем слепок
    data = np.load(file_path)
    if "imprint" not in data:
        print(f"В файле {file_path} нет ключа 'imprint'.")
        return
    voice_imprint = data["imprint"]
    print("Исходная размерность вашего слепка:", voice_imprint.shape)

    # 4. Простейший Linear-мэппинг (если нужно 256, а у вас 64)
    desired_dim = 256
    if voice_imprint.shape[0] != desired_dim:
        print(f"Слепок имеет размер {voice_imprint.shape[0]}, а нужно {desired_dim}.")
        print("Делаем НЕобученную 'прокладку' (Linear) — качество будет низким.")
        in_dim = voice_imprint.shape[0]
        mapper = nn.Linear(in_dim, desired_dim, bias=False)
        with torch.no_grad():
            tensor_64 = torch.from_numpy(voice_imprint).float().unsqueeze(0)  # (1, in_dim)
            tensor_256 = mapper(tensor_64)
        voice_imprint_256 = tensor_256.squeeze(0).numpy()
    else:
        voice_imprint_256 = voice_imprint

    # 5. Готовим выходную папку
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_wav = os.path.join(output_dir, "synthesized.wav")

    # 6. Текст для синтеза (русский)
    text_to_speak = "Привет! Это тестовая синтеза голоса на русском языке."

    # 7. Загружаем модель your_tts
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/your_tts",
        progress_bar=False,
        gpu=False
    )

    # 8. Вызов tts_to_file:
    #    - Обязательно указываем speaker, т.к. модель multi-speaker
    #    - language="ru", раз текст русский
    try:
        tts.tts_to_file(
            text=text_to_speak,
            file_path=output_wav,
            language="ru",
            speaker="my_custom_speaker",         # <-- ВАЖНО: назвать как угодно
            speaker_embedding=voice_imprint_256  # <-- наш кастомный вектор (256)
        )
        print(f"Синтез завершён! Результат: {output_wav}")
    except Exception as ex:
        print("Ошибка при синтезе:", ex)

if __name__ == "__main__":
    main()
