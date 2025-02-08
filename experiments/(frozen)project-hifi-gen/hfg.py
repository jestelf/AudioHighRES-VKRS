import sys
import os
import torch
import torchaudio
import torchaudio.transforms as T
import json
from types import SimpleNamespace

# Указать путь к репозиторию HiFi-GAN
hifi_gan_path = "D:/prhfg/hifi-gan"
sys.path.append(hifi_gan_path)

from models import Generator  # Импортируем Generator из локального HiFi-GAN

# Функция для загрузки HiFi-GAN модели
def load_model(checkpoint_path, config_path):
    # Читаем конфигурацию из config.json
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Преобразуем словарь конфигурации в объект
    config = SimpleNamespace(**config_dict)

    # Инициализируем генератор с использованием объекта конфигурации
    generator = Generator(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator, config

# Функция для преобразования аудио в мел-спектрограмму
def audio_to_mel(audio_path, config):
    waveform, sr = torchaudio.load(audio_path)
    if sr != config.sampling_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=config.sampling_rate)
        waveform = resampler(waveform)
    mel_transform = T.MelSpectrogram(
        sample_rate=config.sampling_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_size,
        n_mels=config.num_mels,
        f_min=config.fmin,
        f_max=config.fmax,
    )
    mel = mel_transform(waveform)
    return mel

def enhance_audio(input_audio, output_audio, model_path, config_path):
    print("Загрузка модели и конфигурации...")
    generator, config = load_model(model_path, config_path)

    print("Преобразование аудио в мел-спектрограмму...")
    mel = audio_to_mel(input_audio, config)

    print("Генерация улучшенного аудио...")
    with torch.no_grad():
        mel = mel.squeeze(0)  # Из [1, 1, num_mels, time_steps] в [1, num_mels, time_steps]
        audio = generator(mel.unsqueeze(0)).squeeze(0).cpu()  # Генерация аудио

    print("Сохранение улучшенного аудио...")
    audio = audio.squeeze()  # Убираем лишнее измерение
    torchaudio.save(output_audio, audio.unsqueeze(0), config.sampling_rate)
    print(f"Улучшенное аудио сохранено в {output_audio}")


# Меню выбора входного аудио
def select_audio_file(directory):
    print("Выберите файл для обработки:")
    audio_files = [f for f in os.listdir(directory) if f.endswith(".wav")]
    if not audio_files:
        print("Нет доступных .wav файлов в указанной папке.")
        return None
    for i, file in enumerate(audio_files):
        print(f"{i + 1}: {file}")
    while True:
        try:
            choice = int(input("Введите номер файла: ")) - 1
            if 0 <= choice < len(audio_files):
                return os.path.join(directory, audio_files[choice])
            else:
                print("Неправильный выбор, попробуйте снова.")
        except ValueError:
            print("Введите корректный номер.")

# Путь к файлам
model_path = "D:/prhfg/LJ_FT_T2_V2/generator"  # Ваш файл модели
config_path = "D:/prhfg/LJ_FT_T2_V2/config.json"  # Файл конфигурации
input_directory = "D:/prhfg"  # Директория с входными файлами
output_audio = "D:/prhfg/output.wav"  # Выходной файл

# Запуск
input_audio = select_audio_file(input_directory)
if input_audio:
    enhance_audio(input_audio, output_audio, model_path, config_path)
else:
    print("Процесс завершён без обработки.")
