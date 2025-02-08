import random
import torch
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

class Tacotron2Logger(SummaryWriter):
    """
    Логгер для отслеживания процесса обучения Tacotron2 с помощью TensorBoard.
    """

    def __init__(self, logdir):
        """
        Инициализация логгера.

        Аргументы:
            logdir (str): путь к директории для сохранения логов TensorBoard.
        """
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration, iteration):
        """
        Логгирование данных во время обучения.

        Аргументы:
            reduced_loss (float): средняя потеря на текущей итерации.
            grad_norm (float): норма градиентов.
            learning_rate (float): текущий learning rate.
            duration (float): время выполнения одной итерации.
            iteration (int): номер текущей итерации.
        """
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        """
        Логгирование данных во время валидации.

        Аргументы:
            reduced_loss (float): средняя потеря на валидации.
            model (torch.nn.Module): текущая модель Tacotron2.
            y (tuple): истинные значения (мел-спектрограммы и gate-таргеты).
            y_pred (tuple): предсказанные значения (мел-спектрограммы, gate-выходы, alignments).
            iteration (int): номер текущей итерации.
        """
        self.add_scalar("validation.loss", reduced_loss, iteration)

        # Распаковываем предсказания модели
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # Логируем распределение параметров модели
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')  # Заменяем точки для корректного отображения в TensorBoard
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # Выбираем случайный индекс примера для визуализации
        idx = random.randint(0, alignments.size(0) - 1)

        # Логируем изображения: alignments, mel-спектрограммы и gate-выходы
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')

        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')

        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')

        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
