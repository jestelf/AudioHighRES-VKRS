from torch import nn

class Tacotron2Loss(nn.Module):
    """
    Функция потерь для модели Tacotron2.
    Состоит из двух компонентов:
      - MSELoss для мел-спектрограмм
      - BCEWithLogitsLoss для предсказания gate-выхода (завершения фразы)
    """

    def __init__(self):
        """
        Инициализация функции потерь.
        """
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        """
        Вычисляет сумму потерь для модели Tacotron2.

        Аргументы:
            model_output (tuple): выходы модели (мел-спектрограммы, gate-выход).
            targets (tuple): истинные значения (мел-спектрограммы и gate-таргеты).

        Возвращает:
            torch.Tensor: сумма потерь (MSE для мел-спектрограмм + BCE для gate-выхода).
        """

        # Истинные значения (целевые мел-спектрограммы и gate-таргеты)
        mel_target, gate_target = targets[0], targets[1]

        # Отключаем градиенты для целевых значений
        mel_target.requires_grad = False
        gate_target.requires_grad = False

        # Преобразуем gate_target в 2D-форму (PyTorch BCE требует размерности [batch, 1])
        gate_target = gate_target.view(-1, 1)

        # Выходы модели
        mel_out, mel_out_postnet, gate_out, _ = model_output

        # Преобразуем gate_out в 2D-форму для сравнения
        gate_out = gate_out.view(-1, 1)

        # MSE Loss для предсказанных мел-спектрограмм (до и после postnet)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                   nn.MSELoss()(mel_out_postnet, mel_target)

        # BCE Loss для предсказаний gate-выхода (сигнала завершения)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        # Итоговая потеря: сумма мел-спектрограммного и gate-компонента
        return mel_loss + gate_loss
