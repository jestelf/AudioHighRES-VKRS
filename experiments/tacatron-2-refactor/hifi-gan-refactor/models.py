# -*- coding: utf-8 -*-

"""
Модуль с реализацией генератора и дискриминаторов (Multi-Period, Multi-Scale) 
для HiFi-GAN (или похожих нейросетевых вокодеров). 
Содержит несколько основных классов:

1) ResBlock1 и ResBlock2 — варианты резидуальных блоков, 
   используемых в генераторе для пропускания сигнала через ряд 
   свёрточных блоков с разными расширенными (dilated) свёртками.

2) Generator — основной генератор, который по входу (обычно мел-спектрограмма) 
   генерирует аудиосигнал. Состоит из серии апсемплингов (ConvTranspose1d) 
   и наборов резидуальных блоков.

3) DiscriminatorP и MultiPeriodDiscriminator — дискриминаторы, работающие 
   с периодическим представлением сигнала (разбиение аудиосигнала на 
   временные фрагменты длиной period, конвертируемые в двумерный вид).

4) DiscriminatorS и MultiScaleDiscriminator — дискриминаторы, работающие 
   с разными масштабами аудиосигнала (через усреднение), 
   также используют набор свёрточных слоёв.

5) Функции для расчёта лоссов: feature_loss, discriminator_loss, generator_loss.

Содержит все необходимые элементы для обучения GAN-подобной архитектуры, 
где генератор обучается против набора дискриминаторов (Multi-Period и Multi-Scale).
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from utils import init_weights, get_padding

LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    """
    ResBlock1 - вариант резидуального блока, использующего 3 пары 
    (конволюция (dilated), конволюция без дополнительного dilation). 
    Каждый блок добавляет skip connection x + xt. 
    """
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        """
        Параметры:
        ----------
        h : объект гиперпараметров (может не использоваться напрямую).
        channels : int
            Количество каналов (фильтров) в свёртках.
        kernel_size : int
            Размер ядра свёртки.
        dilation : tuple of int
            Множители дилатации (расширения) для трёх последовательных блоков.
        """
        super(ResBlock1, self).__init__()
        self.h = h

        # Первый набор свёрток (dilated)
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        # Второй набор свёрток (без dilation=1)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        """
        Прямой проход через резидуальные блоки. Каждый блок:
          - leaky_relu(x)
          - свёртка dilated
          - leaky_relu(выход)
          - свёртка без dilation
          - сложение с исходным x (skip connection).
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """
        Удаляем weight_norm со всех свёрточных слоёв после тренировки 
        (для ускорения инференса).
        """
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    """
    ResBlock2 - упрощённый вариант резидуального блока, 
    использующего два dilated-рез блока в одном проходе.
    """
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        """
        Параметры:
        ----------
        h : объект гиперпараметров.
        channels : int
            Количество каналов свёрток.
        kernel_size : int
            Размер ядра свёртки.
        dilation : tuple of int
            Множители дилатации для свёрток.
        """
        super(ResBlock2, self).__init__()
        self.h = h

        # Свёртки dilated
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        """
        Прямой проход: каждая свёртка обрабатывает x (через leaky_relu),
        затем добавляется skip connection.
        """
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """
        Удаляем weight_norm для ускорения инференса.
        """
        for l in self.convs:
            remove_weight_norm(l)


class Generator(nn.Module):
    """
    Generator — основная часть модели, отвечающая за преобразование 
    входных фич (обычно мел-спектрограммы) в аудиосигнал.
    Использует последовательные апсемплинги (ConvTranspose1d) и 
    серии резидуальных блоков (ResBlock1 или ResBlock2).
    """
    def __init__(self, h):
        """
        Параметры:
        ----------
        h : объект гиперпараметров, содержащий:
            - resblock_kernel_sizes, resblock_dilation_sizes
            - upsample_rates, upsample_kernel_sizes
            - upsample_initial_channel
            - resblock (строка '1' или '2'), указывающая какой ResBlock использовать
        """
        super(Generator, self).__init__()
        self.h = h

        # Количество разновидностей (kernels) резблоков
        self.num_kernels = len(h.resblock_kernel_sizes)
        # Количество апсемплингов
        self.num_upsamples = len(h.upsample_rates)

        # Начальная свёртка (conv_pre), чтобы увеличить кол-во каналов
        self.conv_pre = weight_norm(
            Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # Выбираем какой ResBlock использовать
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        # Блоки апсемплинга (ConvTranspose1d)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            # h.upsample_initial_channel // (2**i) -> текущий размер каналов
            # h.upsample_initial_channel // (2**(i+1)) -> размер на выходе этого апсемплинга
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2 ** i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k, u,
                        padding=(k - u) // 2
                    )
                )
            )

        # Резидуальные блоки после каждого апсемплинга
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        # Финальная свёртка (conv_post) -> один канал (1) на выходе
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        # Инициализация весов
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        """
        Прямой проход через генератор:
        1) conv_pre
        2) несколько апсемплингов с последующими резблоками
        3) conv_post
        4) tanh на выходе
        """
        # Начальная свёртка
        x = self.conv_pre(x)

        # Проходим по всем апсемплингам
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            # Прогон через серию резблоков
            xs = None
            for j in range(self.num_kernels):
                # Каждый апсемплинг имеет num_kernels резблоков
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            # Усредняем выход нескольких резблоков
            x = xs / self.num_kernels

        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """
        Удаляем weight_norm со всех слоёв генератора после обучения.
        """
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(nn.Module):
    """
    DiscriminatorP — периодический дискриминатор, который разбивает сигнал 
    на 2D-тензор формы (B, C, T//period, period) и пропускает через 
    серию 2D-свёрток.
    """
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        """
        Параметры:
        ----------
        period : int
            Период разбиения сигнала (сколько сэмплов в одном фрагменте).
        kernel_size : int
            Размер ядра свёртки по временной оси.
        stride : int
            Шаг (stride) по временной оси.
        use_spectral_norm : bool
            Использовать ли спектральную нормализацию вместо weight_norm.
        """
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        # Набор 2D-свёрток
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), 
                          padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), 
                          padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1),
                          padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1),
                          padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, 
                          padding=(2, 0))),
        ])
        # Финальный свёрточный слой
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        """
        Прямой проход через дискриминатор:
        1) Преобразование 1D сигнала в 2D (с учётом period).
        2) Последовательные свёртки с leaky_relu.
        3) Финальная свёртка, flatten результата для скалярной оценки.

        Возвращает:
        -----------
        x : Tensor
            Выход дискриминатора (B, что-то).
        fmap : list of Tensors
            Список промежуточных фичмап (для feature matching loss).
        """
        fmap = []

        b, c, t = x.shape
        # Если длина не делится на period, дополним нулями (pad)
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad

        # Преобразуем (B, 1, T) -> (B, 1, T//period, period)
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """
    MultiPeriodDiscriminator — набор DiscriminatorP с разными period 
    (2, 3, 5, 7, 11 и т.д.). Предполагается, что на каждом period 
    дискриминатор смотрит на сигнал с разным "шагом".
    """
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        """
        Прямой проход: пропускаем реальный y и сгенерированный y_hat 
        через все периодические дискриминаторы.
        
        Возвращает:
        -----------
        y_d_rs, y_d_gs : списки выходов дискриминаторов (реальный/фейковый)
        fmap_rs, fmap_gs : списки фичмап
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        # Прогон по всем дискриминаторам
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    """
    DiscriminatorS — "Scale"-дискриминатор, 
    который анализирует сигнал в одном масштабе (без периодического разбиения).
    """
    def __init__(self, use_spectral_norm=False):
        """
        Параметры:
        ----------
        use_spectral_norm : bool
            Флаг использования spectral_norm вместо weight_norm.
        """
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        # Набор 1D-свёрток
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        # Финальная свёртка
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        Прямой проход: серия 1D-свёрток с leaky_relu -> conv_post -> flatten.

        Возвращает:
        -----------
        x : Tensor
            Выход дискриминатора.
        fmap : list of Tensor
            Список промежуточных фичмап.
        """
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """
    MultiScaleDiscriminator — набор DiscriminatorS, 
    где сигнал последовательно усредняется (AvgPool1d) для разных масштабов. 
    Используются три дискриминатора:
      1) без усреднения
      2) с одним пулом
      3) с двумя пулами.
    """
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        # Первый дискриминатор со спектральной нормализацией
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        # Два пулера для изменения масштаба
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        """
        Прогон через все скейл-дискриминаторы:
        1) Первый использует исходный сигнал.
        2) Второй — пропущенный через meanpool1.
        3) Третий — через meanpool2 (после первого) или аналогично.

        Возвращает:
        -----------
        y_d_rs, y_d_gs : списки выходов (для реального и сгенерированного)
        fmap_rs, fmap_gs : списки фичмап
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    """
    Подсчёт потерь на уровне фич (feature matching loss).
    Сравнивает промежуточные фичмапы (fmap_r, fmap_g) каждого дискриминатора.

    Параметры:
    ----------
    fmap_r : list of list of Tensors
        Фичмапы для реального сигнала (список дискриминаторов, 
        внутри списка — слои).
    fmap_g : list of list of Tensors
        Фичмапы для сгенерированного сигнала.
    
    Возвращает:
    -----------
    Скаляр (Tensor), сумма l1-разниц по всем фичмапам, 
    умноженная на 2 в конце.
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Подсчёт лосса дискриминатора. Для реальных сэмплов:
    (1 - D(x))^2, для фейковых:
    (D(x_hat))^2 (LSGAN-стиль).
    
    Параметры:
    ----------
    disc_real_outputs : list of Tensor
        Выходы дискриминатора на реальных данных.
    disc_generated_outputs : list of Tensor
        Выходы дискриминатора на сгенерированных данных.

    Возвращает:
    -----------
    loss : скаляр (Tensor)
        Общий лосс дискриминатора (сумма).
    r_losses, g_losses : списки float значений 
        Поэлементные лоссы для каждого дискриминатора.
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """
    Подсчёт лосса генератора по выходам дискриминатора. 
    Генератор стремится к тому, чтобы D(x_hat) было как можно ближе к 1:
    (1 - D(x_hat))^2.

    Параметры:
    ----------
    disc_outputs : list of Tensor
        Выходы всех дискриминаторов на фейковых данных.
    
    Возвращает:
    -----------
    loss : скаляр (Tensor) — сумма лоссов 
    gen_losses : список float значений по каждому дискриминатору.
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
