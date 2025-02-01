import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm

# Используем backend "Agg" для Matplotlib (чтобы избежать проблем с GUI)
matplotlib.use("Agg")
import matplotlib.pylab as plt

def plot_spectrogram(spectrogram):
    """
    Визуализирует мел-спектрограмму и возвращает объект Matplotlib Figure.

    Аргументы:
        spectrogram (np.ndarray или Tensor): спектрограмма для визуализации.

    Возвращает:
        fig (Figure): объект Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Отображаем спектрограмму как изображение
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)

    # Прорисовываем изображение и закрываем фигуру
    fig.canvas.draw()
    plt.close()

    return fig

def init_weights(m, mean=0.0, std=0.01):
    """
    Инициализирует веса свёрточных слоев нормальным распределением.

    Аргументы:
        m (torch.nn.Module): слой нейросети.
        mean (float): среднее значение распределения.
        std (float): стандартное отклонение.
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        m.weight.data.normal_(mean, std)

def apply_weight_norm(m):
    """
    Применяет нормализацию весов к свёрточным слоям модели.

    Аргументы:
        m (torch.nn.Module): слой нейросети.
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        weight_norm(m)  # Применяем weight normalization

def get_padding(kernel_size, dilation=1):
    """
    Вычисляет паддинг для свёрточного слоя с учётом размера ядра и дилатации.

    Аргументы:
        kernel_size (int): размер ядра свёртки.
        dilation (int): коэффициент дилатации.

    Возвращает:
        padding (int): вычисленный паддинг.
    """
    return int((kernel_size * dilation - dilation) / 2)

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
    
    # Загружаем чекпоинт
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    
    return checkpoint_dict

def save_checkpoint(filepath, obj):
    """
    Сохраняет объект (например, модель или её параметры) в файл.

    Аргументы:
        filepath (str): путь для сохранения файла.
        obj (dict или torch.nn.Module): объект для сохранения.
    """
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)  # Сохраняем объект в файл
    print("Complete.")

def scan_checkpoint(cp_dir, prefix):
    """
    Поиск последнего сохранённого чекпоинта в директории.

    Аргументы:
        cp_dir (str): директория, где хранятся чекпоинты.
        prefix (str): префикс имени файла чекпоинта (например, "checkpoint_").

    Возвращает:
        str или None: путь к последнему чекпоинту или None, если файлов нет.
    """
    pattern = os.path.join(cp_dir, prefix + '????????')  # Формируем шаблон поиска
    cp_list = glob.glob(pattern)  # Ищем все файлы, соответствующие шаблону
    
    if len(cp_list) == 0:
        return None
    
    return sorted(cp_list)[-1]  # Возвращаем последний чекпоинт (по возрастанию имён файлов)
