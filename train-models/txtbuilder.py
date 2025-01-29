import os
import pandas as pd

def prepare_vits_data(tsv_path, audio_dir, output_txt):
    """Генерация train.txt или val.txt для VITS."""
    data = pd.read_csv(tsv_path, sep='\t', engine='python')
    with open(output_txt, 'w', encoding='utf-8') as f:
        for _, row in data.iterrows():
            # Путь к аудио
            audio_file = os.path.basename(row['audio_path'])
            audio_path = os.path.join(audio_dir, audio_file)

            # Проверка существования файла
            if not os.path.exists(audio_path):
                print(f"Пропускаем строку: файл {audio_path} не найден")
                continue

            # Текст
            text = row['speaker_text'] if pd.notna(row['speaker_text']) else "Нет текста"
            text = text.strip()

            # Эмоции
            annotator_emo = row['annotator_emo'] if pd.notna(row['annotator_emo']) else "neutral"
            speaker_emo = row['speaker_emo'] if pd.notna(row['speaker_emo']) else "neutral"

            # Идентификатор спикера
            speaker_id = row['source_id'] if pd.notna(row['source_id']) else "0"

            # Запись строки в файл
            f.write(f"{audio_path}|{text}|{speaker_id}|{annotator_emo}|{speaker_emo}\n")

# Пути к данным
TRAIN_TSV = "D:/DatasetDusha/crowd_train/raw_crowd_train.tsv"
TRAIN_AUDIO_DIR = "D:/DatasetDusha/crowd_train/wavs"
VAL_TSV = "D:/DatasetDusha/crowd_test/raw_crowd_test.tsv"
VAL_AUDIO_DIR = "D:/DatasetDusha/crowd_test/wavs"

# Генерация файлов
prepare_vits_data(TRAIN_TSV, TRAIN_AUDIO_DIR, "vits_train.txt")
prepare_vits_data(VAL_TSV, VAL_AUDIO_DIR, "vits_val.txt")
