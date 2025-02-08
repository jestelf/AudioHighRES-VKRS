from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator

# Глобальные переменные для хранения конфигурации и устройства
h = None
device = None

def load_checkpoint(filepath, device):
    """
    Загружает сохранённый чекпоинт модели.

    Аргументы:
        filepath (str): путь к файлу чекпоинта.
        device (str или torch.device): устройство, на которое загружается модель (CPU/GPU).

    Возвращает:
        checkpoint_dict (dict): загруженные параметры модели.
    """
    assert os.path.isfile(filepath), "Файл чекпоинта не найден!"
    print("Loading '{}'".format(filepath))

    # Загружаем параметры модели
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")

    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    """
    Поиск последнего сохранённого чекпоинта в директории.

    Аргументы:
        cp_dir (str): директория, где хранятся чекпоинты.
        prefix (str): префикс имени файла чекпоинта (например, "checkpoint_").

    Возвращает:
        str: путь к последнему чекпоинту или пустая строка, если чекпоинтов нет.
    """
    pattern = os.path.join(cp_dir, prefix + '*')  # Формируем шаблон поиска
    cp_list = glob.glob(pattern)  # Ищем файлы чекпоинтов

    if len(cp_list) == 0:
        return ''

    return sorted(cp_list)[-1]  # Возвращаем последний найденный чекпоинт

def inference(a):
    """
    Выполняет генерацию аудиофайлов из мел-спектрограмм с использованием обученного генератора.

    Аргументы:
        a (Namespace): аргументы командной строки.
    """
    # Инициализируем модель генератора и загружаем веса
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    # Получаем список входных мел-спектрограмм
    filelist = os.listdir(a.input_mels_dir)

    # Создаём директорию для выходных файлов, если её нет
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()

    with torch.no_grad():
        for i, filename in enumerate(filelist):
            # Загружаем мел-спектрограмму (npz или npy формат)
            x = np.load(os.path.join(a.input_mels_dir, filename))
            x = torch.FloatTensor(x).to(device)

            # Запускаем генератор для синтеза аудио
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE  # Обратно денормализуем
            audio = audio.cpu().numpy().astype('int16')  # Преобразуем в 16-битное аудио

            # Сохраняем сгенерированный файл
            output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated_e2e.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)

def main():
    """
    Основная функция, выполняющая запуск процесса инференса (синтеза речи).
    """
    print('Initializing Inference Process..')

    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='test_mel_files', help="Директория с входными мел-спектрограммами")
    parser.add_argument('--output_dir', default='generated_files_from_mel', help="Директория для сохранения результатов")
    parser.add_argument('--checkpoint_file', required=True, help="Путь к файлу с весами модели")
    a = parser.parse_args()

    # Загружаем конфигурацию модели из JSON-файла
    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    # Глобально сохраняем конфигурацию
    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    # Устанавливаем случайное начальное значение (seed) для воспроизводимости
    torch.manual_seed(h.seed)

    # Определяем устройство (GPU или CPU)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Запускаем инференс
    inference(a)

if __name__ == '__main__':
    main()
