#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Сборщик датасета синтезированной речи с использованием модели xtts_v2.

Скрипт:
  - Создает папки datasets/original и datasets/synthesized, если их нет.
  - Загружает LibriSpeech (test-clean) в папку original.
  - Фильтрует записи по длительности (например, от 8 до 12 сек) и числу слов (например, 25–40).
  - Для каждой подходящей записи сохраняет оригинальное аудио во временный файл
    и вызывает модель xtts_v2 для генерации синтезированной речи (используя аудио-референс).
  - Сгенерированное аудио сохраняется в папку synthesized.

Обратите внимание: в данном варианте используется локальный вызов модели xtts_v2,
которая загружается через TTS.api. Если у вас требуется обращаться к сервису по IP, настройте его отдельно.
"""

import os
import sys
import argparse
import random
import tempfile
import torchaudio

from tqdm import tqdm

# Базовая директория
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_dataset_folders():
    base_dataset_dir = os.path.join(BASE_DIR, "datasets")
    original_dir = os.path.join(base_dataset_dir, "original")
    synthesized_dir = os.path.join(base_dataset_dir, "synthesized")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(synthesized_dir, exist_ok=True)
    return original_dir, synthesized_dir

# ---------------------------------------------------
# Функция синтеза с использованием xtts_v2
# ---------------------------------------------------
from TTS.api import TTS
# Загружаем модель xtts_v2 (убедитесь, что модель поддерживает клонирование голоса)
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

def synthesize_speech_xtts(text, reference_audio_path, language="en"):
    """
    Генерирует синтезированное аудио с использованием модели xtts_v2.
    Передает в модель текст, путь к файлу-референсу (speaker_wav) и язык.
    Сохраняет сгенерированное аудио во временный файл, затем возвращает его бинарное содержимое.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            out_path = tmp_out.name
        tts_model.tts_to_file(
            text=text,
            file_path=out_path,
            speaker_wav=reference_audio_path,
            language=language
        )
        with open(out_path, "rb") as f:
            audio_content = f.read()
        os.remove(out_path)
        return audio_content
    except Exception as e:
        print("Ошибка синтеза через xtts_v2:", e)
        return None

# ---------------------------------------------------
# 1) Логирование (если требуется, можно добавить)
# ---------------------------------------------------
def setup_logging(backup_subdir="dataset_collector_backup"):
    backup_dir = os.path.join(BASE_DIR, backup_subdir)
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_backup = os.path.join(backup_dir, timestamp)
    os.makedirs(current_backup, exist_ok=True)
    return current_backup

# ---------------------------------------------------
# Основной сборщик датасета
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Сборщик датасета синтезированной речи с использованием xtts_v2")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Желаемое число синтезированных аудио (по умолчанию 100)")
    parser.add_argument("--min_words", type=int, default=25,
                        help="Минимальное число слов в транскрипте (по умолчанию 25)")
    parser.add_argument("--max_words", type=int, default=40,
                        help="Максимальное число слов в транскрипте (по умолчанию 40)")
    parser.add_argument("--min_duration", type=float, default=8.0,
                        help="Минимальная длительность аудио в секундах (по умолчанию 8)")
    parser.add_argument("--max_duration", type=float, default=12.0,
                        help="Максимальная длительность аудио в секундах (по умолчанию 12)")
    args = parser.parse_args()

    # Создаем папки для датасетов
    original_dir, synthesized_dir = create_dataset_folders()
    print(f"Оригинальный датасет: {original_dir}")
    print(f"Синтезированное аудио будет сохраняться в: {synthesized_dir}")

    # Загружаем LibriSpeech (test-clean)
    libri = torchaudio.datasets.LIBRISPEECH(root=original_dir, url="test-clean", download=True)

    selected_indices = []
    # Фильтруем записи по длительности и числу слов
    for i in range(len(libri)):
        waveform, sample_rate, transcript, _, _, _ = libri[i]
        duration = waveform.shape[1] / sample_rate
        word_count = len(transcript.strip().split())
        if args.min_duration <= duration <= args.max_duration and args.min_words <= word_count <= args.max_words:
            selected_indices.append(i)
        if len(selected_indices) >= args.num_samples:
            break

    if not selected_indices:
        print("Не найдено записей, удовлетворяющих условиям.")
        sys.exit(1)

    print(f"Найдено {len(selected_indices)} записей с длительностью от {args.min_duration} до {args.max_duration} сек и {args.min_words}-{args.max_words} слов.")

    # Создаем временный лог (опционально)
    setup_logging("dataset_collector_backup")

    # Для каждого выбранного индекса генерируем синтезированное аудио
    for idx in tqdm(selected_indices, desc="Синтез аудио"):
        waveform, sample_rate, transcript, _, _, _ = libri[idx]
        # Сохраняем референс аудио во временный файл
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            ref_path = tmp_file.name
        torchaudio.save(ref_path, waveform, sample_rate)
        # Генерируем синтезированное аудио через xtts_v2
        synth_audio = synthesize_speech_xtts(transcript, ref_path, language="en")
        os.remove(ref_path)
        if synth_audio is not None:
            output_path = os.path.join(synthesized_dir, f"synth_{idx}.wav")
            with open(output_path, 'wb') as out_f:
                out_f.write(synth_audio)
            print(f"Сохранено: {output_path}")
        else:
            print(f"Синтез не удался для записи {idx} с текстом:\n{transcript}")

if __name__ == "__main__":
    main()
