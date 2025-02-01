import matplotlib
matplotlib.use("Agg")  # Используем backend Agg, чтобы избежать проблем с графическим интерфейсом
import matplotlib.pylab as plt
import numpy as np

def save_figure_to_numpy(fig):
    """
    Сохраняет matplotlib-рисунок в формате numpy-массива.

    Аргументы:
        fig (matplotlib.figure.Figure): объект фигуры matplotlib.

    Возвращает:
        np.ndarray: массив пикселей изображения (RGB).
    """
    # Получаем данные изображения в формате RGB
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_alignment_to_numpy(alignment, info=None):
    """
    Строит и сохраняет карту выравнивания (alignment) в формате numpy-массива.

    Аргументы:
        alignment (np.ndarray): матрица выравнивания (encoder → decoder).
        info (str, optional): дополнительный текст на графике.

    Возвращает:
        np.ndarray: изображение карты выравнивания.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Отображаем карту выравнивания
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)

    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info  # Добавляем дополнительную информацию, если она есть

    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    # Преобразуем в numpy-массив и закрываем фигуру
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def plot_spectrogram_to_numpy(spectrogram):
    """
    Строит и сохраняет мел-спектрограмму в формате numpy-массива.

    Аргументы:
        spectrogram (np.ndarray): массив значений мел-спектрограммы.

    Возвращает:
        np.ndarray: изображение мел-спектрограммы.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Отображаем мел-спектрограмму
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)

    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    # Преобразуем в numpy-массив и закрываем фигуру
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    """
    Строит и сохраняет выходные значения gate (сигналы окончания) в формате numpy-массива.

    Аргументы:
        gate_targets (np.ndarray): истинные значения gate.
        gate_outputs (np.ndarray): предсказанные значения gate.

    Возвращает:
        np.ndarray: изображение сравнения target vs predicted gate-выходов.
    """
    fig, ax = plt.subplots(figsize=(12, 3))

    # Разбросанные точки: зеленые - целевые, красные - предсказания
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    # Преобразуем в numpy-массив и закрываем фигуру
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data
