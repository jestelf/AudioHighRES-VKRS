import numpy as np
from scipy.io.wavfile import read
import torch

def get_mask_from_lengths(lengths):
    """
    Создает маску для паддинга на основе длин последовательностей.
    
    Аргументы:
        lengths (Tensor): тензор с длинами входных последовательностей.
    
    Возвращает:
        mask (Tensor): булевый тензор [batch, max_len], где True — валидные значения, False — паддинг.
    """
    max_len = torch.max(lengths).item()  # Определяем максимальную длину в батче
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))  # Индексы от 0 до max_len
    mask = (ids < lengths.unsqueeze(1)).bool()  # Создаём булевую маску
    return mask

def load_wav_to_torch(full_path):
    """
    Загружает аудиофайл и конвертирует его в тензор PyTorch.
    
    Аргументы:
        full_path (str): путь к аудиофайлу.
    
    Возвращает:
        data (Tensor): тензор с аудиоданными (float32).
        sampling_rate (int): частота дискретизации аудио.
    """
    sampling_rate, data = read(full_path)  # Читаем wav-файл
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate  # Конвертируем в float32 тензор

def load_filepaths_and_text(filename, split="|"):
    """
    Загружает список файлов и соответствующий им текст.
    
    Аргументы:
        filename (str): путь к файлу, содержащему список путей к аудиофайлам и их текст.
        split (str): разделитель между полями (по умолчанию '|').
    
    Возвращает:
        filepaths_and_text (list): список списков, где каждый элемент — путь к файлу и его текст.
    """
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]  # Разделяем строки по указанному разделителю
    return filepaths_and_text

def to_gpu(x):
    """
    Перемещает тензор на GPU, если он доступен.
    
    Аргументы:
        x (Tensor): входной тензор.
    
    Возвращает:
        Tensor: тензор, размещенный на GPU (если доступен), иначе остается на CPU.
    """
    x = x.contiguous()  # Преобразуем в смежный тензор для оптимального доступа

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)  # Копируем на GPU (не блокируя процесс)
    return torch.autograd.Variable(x)  # Заворачиваем в Variable (для обратного распространения ошибки)
