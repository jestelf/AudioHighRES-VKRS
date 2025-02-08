import os
import shutil
import lmdb
import torchaudio
import pickle
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch

# Параметры
AUDIO_SAMPLE_RATE = 22050  # Частота дискретизации аудио
CHUNK_SIZE = 340000  # Количество записей в одной базе
MAP_SIZE = int(5e10)  # Размер базы данных (50 ГБ)
NUM_WORKERS = 4  # Количество потоков для параллельной обработки
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Используем GPU, если доступно

def process_row(row, data_path):
    """Обработка одной строки данных с использованием GPU."""
    audio_file = os.path.join(data_path, row['audio_path'])
    if not os.path.exists(audio_file):
        return None, f"Файл не найден: {audio_file}"

    try:
        # Загружаем аудио через torchaudio
        audio, sr = torchaudio.load(audio_file)
        audio = audio.to(DEVICE)  # Переносим аудио на GPU
        if sr != AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=AUDIO_SAMPLE_RATE).to(DEVICE)
            audio = resampler(audio)
        audio = audio.squeeze(0).cpu().numpy()  # Преобразуем в numpy массив и возвращаем на CPU
    except Exception as e:
        return None, f"Ошибка при загрузке {audio_file}: {e}"

    sample = {
        "audio": audio,
        "text": row['speaker_text'],
        "emotion": row['annotator_emo'],
        "speaker_id": row['source_id'],
        "duration": row['duration']
    }
    return sample, None

def process_chunk(data_chunk, data_path):
    """Параллельная обработка данных в одном чанке."""
    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_row, row, data_path) for _, row in data_chunk.iterrows()]
        for future in tqdm(futures, desc="Обработка строк", total=len(futures)):
            results.append(future.result())
    return results

def create_lmdb_chunk(data_chunk, data_path, output_lmdb):
    """Создание части LMDB базы."""
    print(f"Создание базы данных: {output_lmdb}")

    if os.path.exists(output_lmdb):
        print(f"Удаление существующей базы: {output_lmdb}")
        shutil.rmtree(output_lmdb, ignore_errors=True)

    # Создаём LMDB
    try:
        lmdb_env = lmdb.open(output_lmdb, map_size=MAP_SIZE, max_dbs=1)
    except lmdb.Error as e:
        print(f"Ошибка при создании LMDB: {e}")
        return

    errors = []
    with lmdb_env.begin(write=True) as txn:
        results = process_chunk(data_chunk, data_path)
        for idx, (result, error) in enumerate(results):
            if error:
                errors.append(error)
                continue
            try:
                txn.put(f"{idx}".encode(), pickle.dumps(result))
            except Exception as e:
                errors.append(f"Ошибка записи {idx}: {e}")

    if errors:
        print(f"Ошибки обработки: {len(errors)}")
        for err in errors[:5]:  # Выводим первые 5 ошибок для проверки
            print(err)

    lmdb_env.close()
    print(f"LMDB dataset сохранён в {output_lmdb}")

def create_lmdb(data_path, tsv_file, output_lmdb_prefix):
    """Создание LMDB базы из TSV файла."""
    tsv_path = os.path.join(data_path, tsv_file)
    print(f"Читаем .tsv файл: {tsv_path}")

    try:
        data = pd.read_csv(tsv_path, sep='\\t', engine='python')
    except Exception as e:
        print(f"Ошибка при чтении файла {tsv_path}: {e}")
        return

    print(f"Загружено {len(data)} записей из {tsv_path}")

    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data.iloc[i:i + CHUNK_SIZE]
        output_lmdb = f"{output_lmdb_prefix}_part_{i // CHUNK_SIZE}.lmdb"
        create_lmdb_chunk(chunk, data_path, output_lmdb)

# Пути к данным
DATASET_PATH_TRAIN = "D:/DatasetDusha/crowd_train"
DATASET_PATH_TEST = "D:/DatasetDusha/crowd_test"

# Настройка количества потоков
NUM_WORKERS = int(input("Введите количество потоков для параллельной обработки (рекомендуется 2-16): "))

# Создание train.lmdb
create_lmdb(
    data_path=DATASET_PATH_TRAIN,
    tsv_file="raw_crowd_train.tsv",
    output_lmdb_prefix="D:/DatasetDusha/train"
)

# Создание test.lmdb
create_lmdb(
    data_path=DATASET_PATH_TEST,
    tsv_file="raw_crowd_test.tsv",
    output_lmdb_prefix="D:/DatasetDusha/test"
)
