import torch
from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression, dynamic_range_decompression
from stft import STFT

class LinearNorm(torch.nn.Module):
    """
    Полносвязный (линейный) слой с инициализацией Xavier.
    Используется для преобразования входных данных в другое пространство признаков.
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        """
        Аргументы:
            in_dim (int): размерность входных данных.
            out_dim (int): размерность выходных данных.
            bias (bool): использовать ли смещение.
            w_init_gain (str): тип инициализации весов (по умолчанию 'linear').
        """
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        # Xavier инициализация для весов слоя
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """
        Прямой проход данных через слой.
        """
        return self.linear_layer(x)

class ConvNorm(torch.nn.Module):
    """
    Сверточный слой с нормализацией. Используется в encoder/decoder.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        """
        Аргументы:
            in_channels (int): количество входных каналов.
            out_channels (int): количество выходных каналов.
            kernel_size (int): размер ядра свертки.
            stride (int): шаг свертки.
            padding (int): дополнение (если не задано, вычисляется автоматически).
            dilation (int): коэффициент дилатации (расширения).
            bias (bool): использовать ли смещение.
            w_init_gain (str): тип инициализации весов.
        """
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1), "kernel_size должен быть нечетным, если padding не указан"
            padding = int(dilation * (kernel_size - 1) / 2)

        # 1D свёрточный слой
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        # Xavier инициализация для весов слоя
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        """
        Прямой проход через сверточный слой.
        """
        conv_signal = self.conv(signal)
        return conv_signal

class TacotronSTFT(torch.nn.Module):
    """
    STFT-преобразование и преобразование в мел-спектрограммы для Tacotron2.
    """

    def __init__(self,
                 filter_length=1024,
                 hop_length=256,
                 win_length=1024,
                 n_mel_channels=80,
                 sampling_rate=22050,
                 mel_fmin=0.0,
                 mel_fmax=8000.0):
        """
        Аргументы:
            filter_length (int): длина окна Фурье-преобразования.
            hop_length (int): шаг сдвига окон STFT.
            win_length (int): длина окна анализа.
            n_mel_channels (int): количество мел-каналов.
            sampling_rate (int): частота дискретизации.
            mel_fmin (float): минимальная частота мел-спектрограммы.
            mel_fmax (float): максимальная частота мел-спектрограммы.
        """
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate

        # STFT-функция для преобразования сигнала в спектрограмму
        self.stft_fn = STFT(filter_length, hop_length, win_length)

        # Создание мел-банков с помощью librosa
        mel_basis = librosa_mel_fn(
            sr=self.sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax
        )
        # Конвертация в тензор PyTorch
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)  # Сохранение в буфере для использования в forward

    def spectral_normalize(self, magnitudes):
        """
        Применяет логарифмическое сжатие динамического диапазона.
        Используется для повышения устойчивости модели к различным уровням громкости.
        """
        return dynamic_range_compression(magnitudes)

    def spectral_de_normalize(self, magnitudes):
        """
        Выполняет обратное преобразование логарифмического сжатия.
        """
        return dynamic_range_decompression(magnitudes)

    def mel_spectrogram(self, y):
        """
        Вычисляет мел-спектрограмму из аудиосигнала.
        
        Аргументы:
            y (Tensor): аудиосигнал, нормализованный в диапазоне [-1, 1].
        
        Возвращает:
            mel_output (Tensor): мел-спектрограмма после нормализации.
        """
        assert(torch.min(y.data) >= -1), "Минимальное значение аудиосигнала должно быть >= -1"
        assert(torch.max(y.data) <= 1), "Максимальное значение аудиосигнала должно быть <= 1"

        # STFT-преобразование аудиосигнала (получаем амплитудный спектр и фазы)
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data

        # Преобразуем амплитудный спектр в мел-спектрограмму
        mel_output = torch.matmul(self.mel_basis, magnitudes)

        # Применяем нормализацию диапазона
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
