import os
import csv
import random
import numpy as np
import torch
import torch.utils.data

from utils import load_wav_to_torch
from text import text_to_sequence
from layers import TacotronSTFT

class TextMelLoader(torch.utils.data.Dataset):
    """
    Класс для загрузки текстовых и мел-спектрограммных данных,
    используемых в нейросетевых моделях синтеза речи.
    """

    def __init__(self, tsv_path, hparams, is_train=True):
        """
        Инициализация загрузчика данных.
        
        Аргументы:
        - tsv_path: путь к TSV-файлу с аннотациями.
        - hparams: гиперпараметры модели.
        - is_train: если True, данные будут перемешаны для тренировки.
        """
        self.tsv_path = tsv_path
        self.is_train = is_train
        self.audiopaths_text_emo_spk = []

        # Читаем TSV-файл с данными
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                ap = row.get("audio_path", "")
                txt = row.get("speaker_text", "")
                emo = row.get("annotator_emo", "")
                spk = row.get("source_id", "")

                if not ap:
                    continue  # Пропускаем пустые строки
                if not txt:
                    txt = " "  # Заполняем пустой текст пробелом
                if not emo:
                    emo = "neutral"  # Если эмоция не указана, ставим "neutral"
                if not spk:
                    spk = "unknown"  # Если нет идентификатора спикера, ставим "unknown"

                # Добавляем данные в список
                self.audiopaths_text_emo_spk.append((ap, txt, emo, spk))

        # Перемешиваем данные при обучении
        if self.is_train:
            random.shuffle(self.audiopaths_text_emo_spk)

        # Создаем STFT-объект для преобразования аудиофайлов в мел-спектрограммы
        self.stft = TacotronSTFT(
            filter_length=hparams.filter_length,
            hop_length=hparams.hop_length,
            win_length=hparams.win_length,
            n_mel_channels=hparams.n_mel_channels,
            sampling_rate=hparams.sampling_rate,
            mel_fmin=hparams.mel_fmin,
            mel_fmax=hparams.mel_fmax
        )
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.hparams = hparams

        # Создаем словари для кодирования спикеров и эмоций
        self.speaker_dict = {}
        self.emotion_dict = {}
        self.spk_count = 1
        self.emo_count = 1
        for ap, txt, em, spk in self.audiopaths_text_emo_spk:
            if spk not in self.speaker_dict:
                self.speaker_dict[spk] = self.spk_count
                self.spk_count += 1
            if em not in self.emotion_dict:
                self.emotion_dict[em] = self.emo_count
                self.emo_count += 1

        # Проверяем, загрузились ли данные
        print(f"[TextMelLoader] Loaded {len(self.audiopaths_text_emo_spk)} entries from {tsv_path}")
        if len(self.audiopaths_text_emo_spk) == 0:
            raise ValueError(f"Dataset is empty! Check your TSV: {tsv_path}")

    def get_mel(self, ap):
        """
        Получает мел-спектрограмму из аудиофайла.
        """
        # Генерируем полный путь к файлу
        root = os.path.dirname(self.tsv_path)
        full_path = os.path.join(root, ap)

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Audio file does not exist: {full_path}")

        # Загружаем аудио и проверяем частоту дискретизации
        audio, sr = load_wav_to_torch(full_path)
        if sr != self.sampling_rate:
            raise ValueError(f"SR mismatch: got {sr}, expected {self.sampling_rate}")

        # Нормализация аудиосигнала
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)  # Добавляем размерность batch

        # Преобразуем аудио в мел-спектрограмму
        melspec = self.stft.mel_spectrogram(audio_norm)
        return melspec.squeeze(0)

    def get_text_seq(self, txt):
        """
        Преобразует текст в последовательность индексов.
        """
        seq = text_to_sequence(txt, self.hparams.text_cleaners)
        return torch.IntTensor(seq)

    def __getitem__(self, index):
        """
        Возвращает один элемент датасета по индексу.
        """
        ap, txt, emo, spk = self.audiopaths_text_emo_spk[index]
        txt_seq = self.get_text_seq(txt)
        mel = self.get_mel(ap)
        spk_id = self.speaker_dict.get(spk, 0)
        emo_id = self.emotion_dict.get(emo, 0)
        return (txt_seq, mel, spk_id, emo_id)

    def __len__(self):
        return len(self.audiopaths_text_emo_spk)


class TextMelCollate:
    """
    Класс для подготовки данных в мини-батчи.
    """

    def __init__(self, n_frames_per_step):
        """
        n_frames_per_step: количество кадров на шаг в модели (обычно 1 или 2).
        """
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """
        Объединяет данные в батч, заполняя их до максимальной длины.
        """
        # Длины входных последовательностей (текст)
        input_lengths = torch.LongTensor([len(x[0]) for x in batch])
        ids_sorted_decreasing = torch.argsort(input_lengths, descending=True)
        max_input_len = input_lengths[ids_sorted_decreasing[0]]

        # Создаем тензоры для текста и метаинформации
        text_padded = torch.zeros(len(batch), max_input_len, dtype=torch.long)
        spk_ids = []
        emo_ids = []

        # Определяем размерность мел-спектрограммы и находим максимальную длину
        num_mels = batch[0][1].size(0)
        max_target_len = max(x[1].size(1) for x in batch)

        # Делаем длину кратной `n_frames_per_step`
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step

        # Создаем тензоры для мел-спектрограммы и маски выхода
        mel_padded = torch.zeros(len(batch), num_mels, max_target_len)
        gate_padded = torch.zeros(len(batch), max_target_len)
        output_lengths = torch.zeros(len(batch), dtype=torch.long)

        # Заполняем тензоры данными
        for i, idx in enumerate(ids_sorted_decreasing):
            txt_seq, mel, spkid, emoid = batch[idx]
            text_padded[i, :txt_seq.size(0)] = txt_seq
            spk_ids.append(spkid)
            emo_ids.append(emoid)
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1  # Указываем конец последовательности
            output_lengths[i] = mel.size(1)

        return (
            text_padded, 
            input_lengths[ids_sorted_decreasing], 
            mel_padded, 
            gate_padded, 
            output_lengths, 
            torch.LongTensor(spk_ids), 
            torch.LongTensor(emo_ids)
        )
