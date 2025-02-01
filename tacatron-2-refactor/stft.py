import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
# Импортируем необходимые функции из librosa
from librosa.util import pad_center as librosa_pad_center
from librosa.util import tiny
from audio_processing import window_sumsquare

class STFT(torch.nn.Module):
    """
    Реализация Кратковременного Фурье-преобразования (STFT) и обратного преобразования (ISTFT).
    Адаптировано из https://github.com/pseeth/pytorch-stft.
    """

    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        """
        Инициализация STFT.

        Аргументы:
            filter_length (int): длина окна Фурье-преобразования.
            hop_length (int): шаг сдвига окон STFT.
            win_length (int): длина окна анализа.
            window (str): тип оконной функции (например, 'hann').
        """
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window

        scale = self.filter_length / self.hop_length

        # Создаем базис Фурье
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int((self.filter_length / 2 + 1))

        # Разделяем вещественную и мнимую части
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        # Создаем forward (STFT) и inverse (ISTFT) матрицы
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        # Если используется оконная функция, применяем ее к базису
        if window is not None:
            assert(self.filter_length >= self.win_length)
            fft_window = get_window(window, self.win_length, fftbins=True)

            # Правильное дополнение окна до нужного размера (исправленный код)
            fft_window = librosa_pad_center(fft_window, size=self.filter_length, axis=0)
            fft_window = torch.from_numpy(fft_window).float()

            forward_basis *= fft_window
            inverse_basis *= fft_window

        # Регистрируем как буферные параметры модели
        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        """
        Выполняет STFT-преобразование входного сигнала.

        Аргументы:
            input_data (Tensor): аудиосигнал размерности [batch, num_samples].

        Возвращает:
            magnitude (Tensor): амплитудный спектр.
            phase (Tensor): фазы.
        """
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # Преобразуем входные данные в формат [batch, 1, num_samples]
        input_data = input_data.view(num_batches, 1, num_samples)

        # Добавляем отраженный паддинг по краям, чтобы избежать артефактов
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect'
        )
        input_data = input_data.squeeze(1)

        # Применяем 1D свёрточное преобразование (аналог FFT)
        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0
        )

        # Отделяем реальные и мнимые части
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        # Вычисляем амплитудный спектр и фазы
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part, real_part)
        return magnitude, phase

    def inverse(self, magnitude, phase):
        """
        Выполняет обратное STFT-преобразование (ISTFT).

        Аргументы:
            magnitude (Tensor): амплитудный спектр.
            phase (Tensor): фазы.

        Возвращает:
            inverse_transform (Tensor): восстановленный аудиосигнал.
        """
        # Воссоздаем комплексные числа из амплитуд и фаз
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase),
             magnitude * torch.sin(phase)],
            dim=1
        )

        # Применяем обратное свёрточное преобразование
        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0
        )

        # Применяем оконную функцию при обратном преобразовании
        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32
            )

            # Индексы ненулевых значений окна
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.from_numpy(window_sum).to(magnitude.device)

            # Коррекция восстановленного сигнала
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            inverse_transform *= float(self.filter_length) / self.hop_length

        # Убираем паддинг
        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2)]
        return inverse_transform

    def forward(self, input_data):
        """
        Выполняет STFT и обратное преобразование (используется для тестирования качества восстановления).

        Аргументы:
            input_data (Tensor): входной аудиосигнал.

        Возвращает:
            reconstruction (Tensor): восстановленный сигнал.
        """
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
