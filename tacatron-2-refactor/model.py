# -*- coding: utf-8 -*-

"""
Модуль model.py
Здесь реализованы основные компоненты Tacotron2:
- LocationLayer
- Attention
- Prenet
- Postnet
- Encoder
- Decoder
- Tacotron2 (объединяет все предыдущие элементы)
"""

from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths


class LocationLayer(nn.Module):
    """
    LocationLayer выполняет обработку прошлых и накопленных весов внимания
    (attention weights), используя свёрточные слои и линейную проекцию.
    Это помогает модели учитывать информацию о том, где мы уже "смотрели"
    в энкодерной последовательности при вычислении внимания.
    """
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        """
        Параметры:
        ----------
        attention_n_filters : int
            Количество каналов в свёрточном слое для location-based attention.
        attention_kernel_size : int
            Размер ядра свёртки для обработки attention weights.
        attention_dim : int
            Размерность проекции для фич внимания.
        """
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        
        # Свёрточный слой, который получает на вход 2 канала:
        # 1) предыдущие веса внимания
        # 2) накопленные (cumulative) веса внимания
        self.location_conv = ConvNorm(
            2, attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1
        )
        
        # Линейный слой, который приводит число каналов
        # (после свертки) к размерности attention_dim
        self.location_dense = LinearNorm(
            attention_n_filters,
            attention_dim,
            bias=False,
            w_init_gain='tanh'
        )

    def forward(self, attention_weights_cat):
        """
        Параметры:
        ----------
        attention_weights_cat : Tensor
            Тензор формы (B, 2, T), где B – размер batch'а,
            2 – два канала весов внимания, T – длина временной оси.
        
        Возвращает:
        -----------
        processed_attention : Tensor
            Тензор формы (B, T, attention_dim). Результат обработки входных
            attention weights через свертку и линейный слой.
        """
        # Пропускаем через свёрточный слой
        processed_attention = self.location_conv(attention_weights_cat)
        
        # Транспонируем с (B, C, T) на (B, T, C),
        # чтобы затем применить линейный слой по последней размерности
        processed_attention = processed_attention.transpose(1, 2)
        
        # Пропускаем через линейный слой
        processed_attention = self.location_dense(processed_attention)
        
        return processed_attention


class Attention(nn.Module):
    """
    Класс Attention реализует механизм внимания. Он принимает
    состояние RNN декодера (query), обработанную энкодером память
    (processed_memory), а также историю весов внимания (attention_weights_cat),
    и вычисляет контекстное векторное представление, а также веса внимания.
    """
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        """
        Параметры:
        ----------
        attention_rnn_dim : int
            Размер скрытого состояния Attention RNN (LSTMCell).
        embedding_dim : int
            Размерность выходов энкодера.
        attention_dim : int
            Размерность скрытого слоя для вычисления энергий внимания.
        attention_location_n_filters : int
            Количество фильтров в свёрточном слое LocationLayer.
        attention_location_kernel_size : int
            Размер ядра свёртки в LocationLayer.
        """
        super(Attention, self).__init__()
        
        # Линейные преобразования для query (скрытое состояние декодера)
        self.query_layer = LinearNorm(
            attention_rnn_dim,
            attention_dim,
            bias=False,
            w_init_gain='tanh'
        )
        
        # Линейное преобразование выходов энкодера
        self.memory_layer = LinearNorm(
            embedding_dim,
            attention_dim,
            bias=False,
            w_init_gain='tanh'
        )
        
        # Финальная линейная проекция для вычисления energy (энергии внимания)
        self.v = LinearNorm(
            attention_dim,
            1,
            bias=False
        )
        
        # LocationLayer для учёта предыдущих и накопленных весов внимания
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim
        )
        
        # Значение, которым мы замаскируем энергию при выходе за границы реальной длины
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        """
        Вычисляет энергии внимания (alignment energies).

        Параметры:
        ----------
        query: Tensor
            Скрытое состояние RNN внимания формы (B, attention_rnn_dim).
        processed_memory: Tensor
            Обработанные энкодерные выходы формы (B, T_enc, attention_dim).
        attention_weights_cat: Tensor
            Предыдущие и накопленные веса внимания формы (B, 2, T_enc).

        Возвращает:
        -----------
        energies: Tensor
            Тензор энергий внимания формы (B, T_enc).
        """
        # Пропускаем query через линейный слой
        processed_query = self.query_layer(query.unsqueeze(1))
        
        # Пропускаем предыдущие и накопленные веса через LocationLayer
        processed_attention_weights = self.location_layer(attention_weights_cat)
        
        # Складываем три слагаемых:
        #   processed_query (B, 1, attention_dim)
        # + processed_attention_weights (B, T_enc, attention_dim)
        # + processed_memory (B, T_enc, attention_dim)
        energies = self.v(
            torch.tanh(
                processed_query + processed_attention_weights + processed_memory
            )
        )
        
        # Убираем последнюю размерность (размер 1 после v)
        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        Основной проход внимания. Возвращает контекстный вектор и новые веса внимания.

        Параметры:
        ----------
        attention_hidden_state : Tensor
            Текущее состояние скрытого слоя RNN внимания (B, attention_rnn_dim).
        memory : Tensor
            Выходы энкодера (B, T_enc, encoder_embedding_dim).
        processed_memory : Tensor
            Обработанные энкодерные выходы (B, T_enc, attention_dim).
        attention_weights_cat : Tensor
            Конкатенация предыдущих и накопленных весов внимания (B, 2, T_enc).
        mask : Tensor или None
            Булевская маска для отсеивания паддинга. Если есть, применяется к энергиям.

        Возвращает:
        -----------
        attention_context : Tensor
            Контекстный вектор, полученный как взвешенная сумма по энкодерным выходам (B, encoder_embedding_dim).
        attention_weights : Tensor
            Новые веса внимания (B, T_enc).
        """
        # Вычисляем энергии внимания (до softmax)
        alignment = self.get_alignment_energies(
            attention_hidden_state,
            processed_memory,
            attention_weights_cat
        )

        # Применяем маску к энергиям внимания, чтобы избежать влияния паддинга
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        # Новые веса внимания, softmax по временной оси
        attention_weights = F.softmax(alignment, dim=1)
        
        # Считаем контекстный вектор через батчевое матричное умножение
        attention_context = torch.bmm(
            attention_weights.unsqueeze(1),  # (B, 1, T_enc)
            memory                             # (B, T_enc, encoder_embedding_dim)
        )
        # Убираем размерность 1
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    """
    Prenet – это несколько последовательных линейных слоёв,
    каждый из которых идёт после активации ReLU и дропаута.
    Обычно используется в декодере перед основными RNN-блоками,
    чтобы способствовать плавному обучению.
    """
    def __init__(self, in_dim, sizes):
        """
        Параметры:
        ----------
        in_dim : int
            Размерность входа.
        sizes : list of int
            Размерности выходов каждого линейного слоя в пренете.
        """
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        
        # Создаём несколько линейных слоёв, каждый – LinearNorm
        self.layers = nn.ModuleList([
            LinearNorm(in_size, out_size, bias=False)
            for (in_size, out_size) in zip(in_sizes, sizes)
        ])

    def forward(self, x):
        """
        Прямой проход через Prenet.
        Применяем ReLU + Dropout к каждому слою.
        """
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """
    Postnet – это блок нескольких 1-D свёрток:
    Обычно 5 свёрточных слоёв, где первые 4 имеют tanh-активацию,
    а последний – линейную (w_init_gain='linear').
    Он корректирует выходные мел-спектрограммы для улучшения качества.
    """
    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        # Первый свёрточный слой: (n_mel_channels -> postnet_embedding_dim)
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    hparams.n_mel_channels,
                    hparams.postnet_embedding_dim,
                    kernel_size=hparams.postnet_kernel_size,
                    stride=1,
                    padding=int((hparams.postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain='tanh'
                ),
                nn.BatchNorm1d(hparams.postnet_embedding_dim)
            )
        )

        # Промежуточные свёрточные слои (postnet_embedding_dim -> postnet_embedding_dim)
        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        hparams.postnet_embedding_dim,
                        hparams.postnet_embedding_dim,
                        kernel_size=hparams.postnet_kernel_size,
                        stride=1,
                        padding=int((hparams.postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain='tanh'
                    ),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim)
                )
            )

        # Последний слой: (postnet_embedding_dim -> n_mel_channels)
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    hparams.postnet_embedding_dim,
                    hparams.n_mel_channels,
                    kernel_size=hparams.postnet_kernel_size,
                    stride=1,
                    padding=int((hparams.postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain='linear'
                ),
                nn.BatchNorm1d(hparams.n_mel_channels)
            )
        )

    def forward(self, x):
        """
        Пропускаем входной мел-спектр через несколько свёрточных слоёв
        с tanh-активацией (для первых слоёв) и Dropout,
        а затем последний слой без tanh (линейный выход).
        """
        # Для всех слоёв, кроме последнего
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        
        # Последний слой (без tanh, но с Dropout)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x


class Encoder(nn.Module):
    """
    Encoder принимает на вход закодированную последовательность символов
    (через embedding), обрабатывает её несколькими свёрточными слоями,
    а затем двунаправленной LSTM (bi-LSTM).
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        convs = []
        # Создаём несколько свёрточных слоёв подряд
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    hparams.encoder_embedding_dim,
                    hparams.encoder_embedding_dim,
                    kernel_size=hparams.encoder_kernel_size,
                    stride=1,
                    padding=int((hparams.encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain='relu'
                ),
                nn.BatchNorm1d(hparams.encoder_embedding_dim)
            )
            convs.append(conv_layer)
        
        self.convolutions = nn.ModuleList(convs)
        
        # Двунаправленная LSTM, уменьшает выходную размерность в 2 раза
        # (так как bidirectional => суммарно размер удваивается)
        self.lstm = nn.LSTM(
            hparams.encoder_embedding_dim,
            int(hparams.encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x, input_lengths):
        """
        Параметры:
        ----------
        x : Tensor
            Энкодерный вход (B, embedding_dim, T_enc).
        input_lengths : Tensor
            Длины реальных входных последовательностей (без паддинга).

        Возвращает:
        -----------
        outputs : Tensor
            Выходы энкодера (B, T_enc, encoder_embedding_dim).
        """
        # Прогоняем через свёрточные слои (каждый со своим BatchNorm и ReLU)
        for conv in self.convolutions:
            x = F.relu(conv(x))
            x = F.dropout(x, 0.5, self.training)
        
        # Транспонируем в (B, T_enc, embedding_dim)
        x = x.transpose(1, 2)
        
        # Пакуем последовательность для LSTM, чтобы избежать
        # вычислений на паддинговых фреймах
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        # Прогоняем через bi-LSTM
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        
        # Распаковываем
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs

    def inference(self, x):
        """
        Используется при инференсе, когда мы не знаем точных длин,
        или не хотим упаковывать/распаковывать.
        """
        # Аналогично forward, но без pack_padded_sequence
        for conv in self.convolutions:
            x = F.relu(conv(x))
            x = F.dropout(x, 0.5, self.training)
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs


class Decoder(nn.Module):
    """
    Decoder – состоит из двух LSTMCell:
    1) Attention RNN (attention_rnn)
    2) Decoder RNN (decoder_rnn)

    Он шаг за шагом генерирует мел-спектрограммы, используя механизм внимания,
    и останавливается, если преодолен порог gate (или достигнут максимум шагов).
    """
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        
        # Основные гиперпараметры
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        # Prenet – обрабатывает входной мел-фрейм (или Go Frame)
        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim]
        )

        # Первый LSTMCell (Attention RNN)
        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim
        )

        # Механизм внимания
        self.attention_layer = Attention(
            hparams.attention_rnn_dim,
            hparams.encoder_embedding_dim,
            hparams.attention_dim,
            hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size
        )

        # Второй LSTMCell (Decoder RNN)
        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1
        )

        # Линейная проекция для вывода мел-спектра
        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step
        )

        # Линейная проекция для gate (сигнал конца)
        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            1,
            bias=True,
            w_init_gain='sigmoid'
        )

    def get_go_frame(self, memory):
        """
        Возвращает нулевой фрейм размера (B, n_mel_channels * n_frames_per_step),
        который используется как первый вход в декодер (Go Frame).
        """
        B = memory.size(0)
        decoder_input = Variable(
            memory.data.new(
                B, self.n_mel_channels * self.n_frames_per_step
            ).zero_()
        )
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """
        Инициализирует состояния RNN и Attention:

        - attention_hidden, attention_cell
        - decoder_hidden, decoder_cell
        - attention_weights, attention_weights_cum, attention_context
        - сохраняет memory и processed_memory (для быстрого доступа)
        - маску (mask) для внимания

        Параметры:
        ----------
        memory : Tensor
            Выходы энкодера (B, T_enc, encoder_embedding_dim).
        mask : Tensor или None
            Если не None, это бинарная маска для отсеивания паддинга при внимании.
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(
            memory.data.new(B, self.attention_rnn_dim).zero_()
        )
        self.attention_cell = Variable(
            memory.data.new(B, self.attention_rnn_dim).zero_()
        )

        self.decoder_hidden = Variable(
            memory.data.new(B, self.decoder_rnn_dim).zero_()
        )
        self.decoder_cell = Variable(
            memory.data.new(B, self.decoder_rnn_dim).zero_()
        )

        self.attention_weights = Variable(
            memory.data.new(B, MAX_TIME).zero_()
        )
        self.attention_weights_cum = Variable(
            memory.data.new(B, MAX_TIME).zero_()
        )
        self.attention_context = Variable(
            memory.data.new(B, self.encoder_embedding_dim).zero_()
        )

        self.memory = memory
        # Обработанные энкодерные выходы для внимания
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """
        Подготавливает декодерные входы (teacher forcing).

        Изначально mel_padded имеет форму (B, n_mel_channels, T_out).
        1) Транспонируем его, чтобы временная размерность была в середине: (B, T_out, n_mel_channels).
        2) Объединяем мел-фреймы по n_frames_per_step: (B, T_out/frames_per_step, n_mel_channels*frames_per_step).
        3) Транспонируем в (T_out/frames_per_step, B, n_mel_channels*frames_per_step),
           чтобы удобно итерироваться во временном цикле.

        Возвращает:
        -----------
        decoder_inputs: Tensor
            Тензор для пренета формы (T_dec, B, n_mel_channels*n_frames_per_step).
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        
        # Объединяем фреймы
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step),
            -1
        )
        
        # (B, T_dec, n_mel_channels*frames_per_step) -> (T_dec, B, ...)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """
        Преобразует выход декодера в удобные формы:

        - Склеивает выходы по временной оси.
        - Транспонирует обратно в (B, n_mel_channels, T_out).
        - Аналогично для gate outputs и alignments.
        """
        # Собираем alignments: (T_out, B, T_enc) -> (B, T_out, T_enc)
        alignments = torch.stack(alignments).transpose(0, 1)
        
        # Собираем gate: (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()
        
        # Собираем mel: (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()

        # Возвращаемся от склейки кадров к изначальному формату
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0),
            -1,
            self.n_mel_channels
        )
        
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """
        Производит один шаг декодера:
        1) Attention RNN
        2) Вычисление внимания и контекста
        3) Decoder RNN
        4) Линейная проекция на мел и gate.

        Параметры:
        ----------
        decoder_input : Tensor
            Вход пренета (B, prenet_dim).

        Возвращает:
        -----------
        decoder_output : Tensor
            Выход мел-спектра (B, n_mel_channels*n_frames_per_step).
        gate_prediction : Tensor
            Выход gate (B, 1).
        attention_weights : Tensor
            Веса внимания (B, T_enc).
        """
        # Объединяем (decoder_input + context) для Attention RNN
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        
        # Attention RNN шаг
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )
        
        # Дропаут к attention_hidden
        self.attention_hidden = F.dropout(
            self.attention_hidden,
            self.p_attention_dropout,
            self.training
        )

        # Подготавливаем веса внимания (предыдущие и накопленные)
        attention_weights_cat = torch.cat(
            (
                self.attention_weights.unsqueeze(1),
                self.attention_weights_cum.unsqueeze(1)
            ),
            dim=1
        )

        # Считаем новый контекст и новые веса внимания
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask
        )

        # Обновляем накопленные веса
        self.attention_weights_cum += self.attention_weights
        
        # Прогоняем через Decoder RNN
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1
        )
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )
        
        # Дропаут к decoder_hidden
        self.decoder_hidden = F.dropout(
            self.decoder_hidden,
            self.p_decoder_dropout,
            self.training
        )

        # Склеиваем decoder_hidden и контекст
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1
        )

        # Линейная проекция на мел (n_mel_channels * frames_per_step)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        # Линейная проекция на gate
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """
        Прямой проход декодера при тренировке (teacher forcing).

        Параметры:
        ----------
        memory : Tensor
            Выходы энкодера (B, T_enc, encoder_embedding_dim).
        decoder_inputs : Tensor
            Учительские мел-спектры (B, n_mel_channels, T_out).
        memory_lengths : Tensor
            Реальные длины выходов энкодера, чтобы создать маску внимания.

        Возвращает:
        -----------
        mel_outputs, gate_outputs, alignments
        """
        # Берём нулевой Go Frame и добавляем его первым
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        
        # Парсим входы для декодера (teacher forcing)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        
        # Конкатенируем Go Frame и остальные фреймы
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        
        # Пропускаем через Prenet
        decoder_inputs = self.prenet(decoder_inputs)

        # Инициализируем состояния декодера
        self.initialize_decoder_states(
            memory,
            mask=~get_mask_from_lengths(memory_lengths)  # инвертируем маску
        )

        mel_outputs, gate_outputs, alignments = [], [], []

        # На каждом временном шаге извлекаем вход из decoder_inputs
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            current_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(current_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        # Собираем все выходы в нужные формы
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """
        Инференс (генерация) без teacher forcing.
        На каждом шаге используем предыдущий выход декодера как вход.

        Параметры:
        ----------
        memory : Tensor
            Выход энкодера (B, T_enc, encoder_embedding_dim).

        Возвращает:
        -----------
        mel_outputs, gate_outputs, alignments
        """
        # Начинаем с нулевого Go Frame
        decoder_input = self.get_go_frame(memory)

        # Инициализируем состояния
        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []

        while True:
            # Пропускаем вход через Prenet
            decoder_input = self.prenet(decoder_input)

            # Декодируем один шаг
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            # Проверяем условие остановки по gate
            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            # Следующий вход – предыдущий выход
            decoder_input = mel_output

        # Собираем выходы в нужную форму
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )
        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    """
    Общая модель Tacotron2, объединяющая:
    1) Embedding (символов)
    2) Encoder
    3) Decoder
    4) Postnet
    5) Плюс дополнительные эмбеддинги спикера и эмоции (spk_embed, emo_embed)
    """
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        # Эмбеддинг для входных символов
        self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
        
        # Инициализация весов эмбеддинга символов
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)

        # Эмбеддинги для спикера и эмоции
        self.spk_embed = nn.Embedding(hparams.n_speakers, hparams.speaker_embedding_dim)
        nn.init.xavier_uniform_(self.spk_embed.weight)
        self.emo_embed = nn.Embedding(hparams.n_emotions, hparams.emotion_embedding_dim)
        nn.init.xavier_uniform_(self.emo_embed.weight)

        # Encoder, Decoder, Postnet
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        """
        Извлекает тензоры из batch и перемещает на нужное устройство (GPU при наличии).

        Параметры:
        ----------
        batch: tuple
            (text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spk_ids, emo_ids)

        Возвращает:
        -----------
        x: tuple
            То же, что и вход, но на GPU (при использовании).
        y: tuple
            (mel_padded, gate_padded), целевые выходы для лосса.
        """
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spk_ids, emo_ids = batch

        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        spk_ids = to_gpu(spk_ids).long()
        emo_ids = to_gpu(emo_ids).long()

        x = (text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spk_ids, emo_ids)
        y = (mel_padded, gate_padded)
        return x, y

    def forward(self, inputs):
        """
        Прямой проход всей модели (Tacotron2):
        1) Эмбеддинг символов -> Encoder
        2) Добавляем эмбеддинг спикера и эмоции к выходам энкодера
        3) Идём в Decoder (с teacher forcing)
        4) Пропускаем результат через Postnet

        Параметры:
        ----------
        inputs: tuple
            (text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spk_ids, emo_ids)

        Возвращает:
        -----------
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments
        """
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spk_ids, emo_ids = inputs
        
        # Эмбеддинг символов
        embedded_text = self.embedding(text_padded).transpose(1, 2)  # (B, embed_dim, T_enc)
        
        # Пропускаем через Encoder
        encoder_outputs = self.encoder(embedded_text, input_lengths)

        # Получаем эмбеддинги спикера и эмоции
        spk_e = self.spk_embed(spk_ids)  # (B, speaker_embedding_dim)
        emo_e = self.emo_embed(emo_ids)  # (B, emotion_embedding_dim)
        
        # Конкатенируем их
        style_cat = torch.cat([spk_e, emo_e], dim=1)  # (B, spk_emb_dim + emo_emb_dim)

        # Расширяем по временной оси, чтобы сложить к encoder_outputs
        B, T_enc, E_enc = encoder_outputs.shape
        sty = style_cat.unsqueeze(1).expand(B, T_enc, style_cat.size(1))
        
        # Дополняем выходы энкодера style-эмбеддингом
        encoder_outputs = torch.cat([encoder_outputs, sty], dim=2)
        
        # Запускаем Decoder
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mel_padded, input_lengths)
        
        # Пропускаем через Postnet (коррекция мел-спектрограммы)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        # Применяем маскирование при необходимости
        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths
        )

    def parse_output(self, outputs, output_lengths=None):
        """
        Если mask_padding = True, применяем маску к выходным мел-спектрам и gate,
        чтобы обнулить значения, соответствующие паддингу.
        """
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            # Маскируем мел-спектры
            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            # Маскируем gate
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)
        return outputs

    def inference(self, inputs, spk_id, emo_id):
        """
        Инференс всей модели. На вход подаются:
        1) Тензор символов (inputs)
        2) spk_id
        3) emo_id

        Возвращает:
        -----------
        mel_out, mel_post, gate_out, align
        """
        # Эмбеддинг символов
        embedded_text = self.embedding(inputs).transpose(1, 2)
        
        # Энкодер
        encoder_outputs = self.encoder.inference(embedded_text)

        # Добавляем стиль (спикер + эмоция)
        B, T_enc, E_enc = encoder_outputs.shape
        spk_e = self.spk_embed(spk_id)  # (B, spk_emb_dim)
        emo_e = self.emo_embed(emo_id)  # (B, emo_emb_dim)
        style_cat = torch.cat([spk_e, emo_e], dim=1)
        style_cat = style_cat.unsqueeze(1).expand(B, T_enc, style_cat.size(1))

        # Дополняем выходы энкодера
        encoder_outputs = torch.cat([encoder_outputs, style_cat], dim=2)

        # Декодер (инференс)
        mel_out, gate_out, align = self.decoder.inference(encoder_outputs)

        # Postnet
        mel_post = self.postnet(mel_out)
        mel_post = mel_out + mel_post

        # parse_output на случай, если требуется маскирование (в инференсе обычно нет паддинга)
        return self.parse_output([mel_out, mel_post, gate_out, align])
