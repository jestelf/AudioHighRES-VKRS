# Импорт необходимых библиотек
import torch
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    Вычисляет сумму квадратов окна для оценки модуляционных эффектов,
    вызванных оконным преобразованием в STFT.
    
    Параметры:
    ----------
    window : string, tuple, number, callable, or list-like
        Спецификация окна, совместимая с `get_window`.
    
    n_frames : int > 0
        Количество анализируемых кадров.
    
    hop_length : int > 0
        Смещение между кадрами (шаг анализа).
    
    win_length : int, optional
        Длина оконной функции. По умолчанию совпадает с `n_fft`.
    
    n_fft : int > 0
        Длина каждого анализируемого кадра.
    
    dtype : np.dtype
        Тип данных выходного массива.
    
    norm : None or str
        Норма для нормализации окна.
    
    Возвращает:
    ----------
    wss : np.ndarray
        Массив с суммой квадратов оконной функции.
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Вычисляем квадрат окна с нормализацией
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Заполняем массив суммой квадратов оконной функции
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    
    return x


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    Алгоритм Гриффина-Лима для фазовой реконкилизации из амплитуд спектрограммы.
    Используется для восстановления аудиосигнала из его спектрального представления.
    
    Параметры:
    ----------
    magnitudes : torch.Tensor
        Спектрограммы амплитуд (без фазы).
    
    stft_fn : объект STFT
        Класс STFT с методами преобразования (STFT) и обратного преобразования (ISTFT).
    
    n_iters : int
        Количество итераций для обновления фазы.
    
    Возвращает:
    ----------
    signal : torch.Tensor
        Восстановленный аудиосигнал.
    """
    # Инициализируем случайные углы (фазы)
    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    
    # Первоначальное восстановление сигнала
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    # Итеративное уточнение фазового спектра
    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)  # Получаем новые фазы из STFT
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)  # Восстанавливаем сигнал
    
    return signal


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    Логарифмическое сжатие динамического диапазона для нормализации амплитуды.
    Часто используется в аудиопредобработке для обучения нейросетей.
    
    Параметры:
    ----------
    x : torch.Tensor
        Входной аудиосигнал или спектрограмма.
    
    C : float
        Коэффициент сжатия.
    
    clip_val : float
        Минимальное значение, чтобы избежать логарифма от нуля.
    
    Возвращает:
    ----------
    torch.Tensor
        Сжатый сигнал.
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    Обратное логарифмическое сжатие для восстановления динамического диапазона.
    
    Параметры:
    ----------
    x : torch.Tensor
        Сжатый входной сигнал.
    
    C : float
        Коэффициент, использованный при сжатии.
    
    Возвращает:
    ----------
    torch.Tensor
        Восстановленный сигнал.
    """
    return torch.exp(x) / C
